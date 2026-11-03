"""Gaussian mask diffusion with explicit semantic-to-state representations.

The cosine schedule is shared with the historical GaussianDiffusion.  The
ordinal representation changes the clean random variable while keeping the
same Gaussian forward process and deterministic DDIM reverse update.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .diffusion import GaussianDiffusion


class MaskStateCodec:
    """Map semantic probabilities to signed one-hot or cumulative states.

    ``class_order`` lists class IDs from absent material to thick material.
    A permuted order provides a control with identical ordinal dimensions.
    Monotonicity is imposed only on predicted clean states.  Noisy states are
    unrestricted, and neighboring pixels have no imposed ordering.
    """

    def __init__(self, representation='ordinal', classes=4, class_order=None):
        if representation not in ('onehot', 'ordinal'):
            raise ValueError('representation must be onehot or ordinal')
        if not isinstance(classes, int) or isinstance(classes, bool) or classes < 2:
            raise ValueError('classes must be an integer greater than one')
        order = tuple(range(classes) if class_order is None else class_order)
        if any(type(value) is not int for value in order) or len(order) != classes or set(order) != set(range(classes)):
            raise ValueError('class_order must be a permutation of semantic class IDs')
        if representation == 'onehot' and order != tuple(range(classes)):
            raise ValueError('class_order controls only the ordinal representation')
        self.representation = representation
        self.classes = classes
        self.class_order = order
        self.channels = classes if representation == 'onehot' else classes - 1

    def from_probabilities(self, probabilities: Tensor) -> Tensor:
        if probabilities.ndim != 4 or probabilities.shape[1] != self.classes:
            raise ValueError('Probabilities must have shape [B, classes, H, W]')
        if self.representation == 'onehot':
            return 2 * probabilities - 1
        ordered = probabilities[:, self.class_order]
        # Explicit sums retain the existing project's deterministic CUDA policy.
        cumulative = torch.stack(
            [ordered[:, threshold:].sum(1) for threshold in range(1, self.classes)],
            dim=1,
        )
        return 2 * cumulative - 1

    def encode(self, mask: Tensor) -> Tensor:
        if mask.ndim != 3 or mask.dtype != torch.long:
            raise ValueError('Masks must be int64 tensors with shape [B, H, W]')
        probabilities = F.one_hot(mask, self.classes).permute(0, 3, 1, 2).float()
        return self.from_probabilities(probabilities)


class GaussianMaskDiffusion(GaussianDiffusion):
    """Clean-mask prediction with a shared training/sampling state mapping."""

    def __init__(self, codec: MaskStateCodec, timesteps=1000, device='cpu'):
        if not isinstance(timesteps, int) or isinstance(timesteps, bool) or timesteps < 2:
            raise ValueError('timesteps must be an integer of at least two')
        super().__init__(timesteps=timesteps, device=device)
        self.codec = codec

    def encode(self, mask: Tensor) -> Tensor:
        return self.codec.encode(mask)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        self._check_state(x0)
        if noise.shape != x0.shape or noise.device != x0.device:
            raise ValueError('Noise and clean state must have matching shapes and devices')
        if t.shape != (len(x0),) or t.dtype != torch.long:
            raise ValueError('Time indices must be int64 with shape [B]')
        if t.device != x0.device or self.abar.device != x0.device:
            raise ValueError('State, times and schedule must share a device')
        if bool(((t < 0) | (t >= self.timesteps)).any()):
            raise ValueError('Time index outside the diffusion schedule')
        return super().q_sample(x0, t, noise)

    def _check_state(self, state: Tensor):
        if state.ndim != 4 or state.shape[1] != self.codec.channels:
            raise ValueError('State has an incompatible number of channels')
        if not state.is_floating_point():
            raise ValueError('Diffusion states must be floating point')

    def loss(
        self,
        logits: Tensor,
        mask: Tensor,
        ce_weight=.5,
        dice_weight=.5,
        reconstruction_weight=1.,
    ) -> dict[str, Tensor]:
        """CE/Dice use semantic probabilities; MSE uses the encoded clean state.

        MSE is averaged over channels in both representations.  Their channel
        counts and pairwise code distances remain explicit experimental factors.
        """
        weights = (ce_weight, dice_weight, reconstruction_weight)
        if any(weight < 0 or not math.isfinite(weight) for weight in weights):
            raise ValueError('Loss weights must be finite and nonnegative')
        probabilities = logits.softmax(1)
        truth = F.one_hot(mask, self.codec.classes).permute(0, 3, 1, 2).float()
        dimensions = (0, 2, 3)
        dice = (2 * (probabilities * truth).sum(dimensions) + 1e-6) / (
            probabilities.sum(dimensions) + truth.sum(dimensions) + 1e-6
        )
        terms = {
            'loss_ce': F.cross_entropy(logits, mask, reduction='none').mean(),
            'loss_dice_fg': 1 - dice[1:].mean(),
            'loss_x0_mse': F.mse_loss(
                self.codec.from_probabilities(probabilities), self.encode(mask)
            ),
        }
        terms['loss_total'] = (
            ce_weight * terms['loss_ce']
            + dice_weight * terms['loss_dice_fg']
            + reconstruction_weight * terms['loss_x0_mse']
        )
        return terms

    def ddim_step(self, state: Tensor, predicted_clean: Tensor, t: int, next_t: int) -> Tensor:
        """Advance between two indices of the same forward variance schedule."""
        if not 0 <= next_t < t < self.timesteps:
            raise ValueError('DDIM requires 0 <= next_t < t < timesteps')
        alpha, next_alpha = self.abar[t], self.abar[next_t]
        predicted_noise = (state - alpha.sqrt() * predicted_clean) / (1 - alpha).sqrt()
        return next_alpha.sqrt() * predicted_clean + (1 - next_alpha).sqrt() * predicted_noise

    @torch.no_grad()
    def sample(self, model, condition, noise: Tensor, steps=20) -> Tensor:
        """Start at Gaussian noise and return four-class semantic probabilities.

        The caller prepares image conditions once.  No target mask is accepted.
        One-step sampling evaluates the maximum-noise endpoint directly.
        """
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= self.timesteps:
            raise ValueError('steps must be an integer between one and timesteps')
        self._check_state(noise)
        if noise.device != self.abar.device:
            raise ValueError('Noise and schedule must share a device')
        times = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
        state = noise.clone()
        for index, time in enumerate(times):
            batch_time = torch.full((len(state),), time, dtype=torch.long, device=state.device)
            logits = model.denoise(condition, state, batch_time)
            probabilities = logits.softmax(1)
            clean = self.codec.from_probabilities(probabilities)
            if clean.shape != state.shape:
                raise ValueError('Denoiser output and noisy state spatial shapes differ')
            if index == len(times) - 1:
                if not bool(torch.isfinite(probabilities).all()):
                    raise FloatingPointError('Nonfinite semantic probabilities from the denoiser')
                return probabilities
            state = self.ddim_step(state, clean, time, times[index + 1])
        raise RuntimeError('Empty diffusion schedule')

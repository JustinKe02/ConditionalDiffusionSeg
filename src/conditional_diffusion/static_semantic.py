"""Static RGB semantics with matched direct, deterministic and diffusion modes.

This is a new, end-to-end platform. It does not load old weights or references.
S0 owns only the RGB encoder, decoder and head. S1/S2 additionally own identical
state/time branches. The shared and state initialization domains independently
use the same registered training seed; constructing a model preserves the
caller's CPU/CUDA random-generator states.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, channels), channels)


class _ResidualBlock(nn.Module):
    """Two pre-activation convolutions, optional additive time projection."""

    def __init__(self, cin: int, cout: int, time_dim: int = 0):
        super().__init__()
        self.first = nn.Sequential(
            _norm(cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1)
        )
        self.second = nn.Sequential(
            _norm(cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1)
        )
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.time = nn.Linear(time_dim, cout) if time_dim else None

    def forward(self, x: Tensor, time: Tensor | None = None) -> Tensor:
        h = self.first(x)
        if self.time is not None:
            if time is None:
                raise ValueError('A state residual block requires a time embedding')
            h = h + self.time(F.silu(time))[:, :, None, None]
        return self.skip(x) + self.second(h)


class _PyramidEncoder(nn.Module):
    """Four spatial scales, two residual blocks per scale."""

    def __init__(self, cin: int, widths: Sequence[int], time_dim: int = 0):
        super().__init__()
        self.input = nn.Conv2d(cin, widths[0], 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                _ResidualBlock(w, w, time_dim),
                _ResidualBlock(w, w, time_dim),
            ])
            for w in widths
        ])
        self.down = nn.ModuleList([
            nn.Conv2d(a, b, 3, stride=2, padding=1)
            for a, b in zip(widths, widths[1:])
        ])

    def forward(self, x: Tensor, time: Tensor | None = None) -> tuple[Tensor, ...]:
        h = self.input(x)
        pyramid = []
        for i, blocks in enumerate(self.blocks):
            for block in blocks:
                h = block(h, time)
            pyramid.append(h)
            if i < len(self.down):
                h = self.down[i](h)
        return tuple(pyramid)


class _HighResolutionDecoder(nn.Module):
    """Shared three-stage decoder; time enters through the state encoder only."""

    def __init__(self, widths: Sequence[int]):
        super().__init__()
        self.up = nn.ModuleList([
            nn.Conv2d(widths[i + 1], widths[i], 3, padding=1)
            for i in (2, 1, 0)
        ])
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                _ResidualBlock(2 * widths[i], widths[i]),
                _ResidualBlock(widths[i], widths[i]),
            ])
            for i in (2, 1, 0)
        ])

    def forward(self, pyramid: Sequence[Tensor]) -> Tensor:
        h = pyramid[-1]
        for up, blocks, skip in zip(self.up, self.blocks, reversed(pyramid[:-1])):
            h = up(F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False))
            h = torch.cat((h, skip), dim=1)
            for block in blocks:
                h = block(h)
        return h


class StaticSemanticModel(nn.Module):
    """S0 direct / S1 deterministic five-step / S2 conditional diffusion.

    ``encode_rgb`` returns an ordinary tuple of tensors, retaining gradients in
    training. ``forward_with_rgb`` executes one state-conditioned decode (or the
    direct decode in S0), so callers may reuse the same tuple over multiple steps.
    ``forward`` executes all five connected refinement steps for S1 and a single
    decode for S0/S2. S2's raw output is used by CE/Dice/MSE; clipping is performed
    by the teacher/sampler, never by the training forward.
    """

    rgb_widths = (32, 64, 128, 256)
    state_widths = (16, 32, 64, 128)
    classes = 4
    time_encoding_dim = 32
    time_embedding_dim = 128

    def __init__(
        self,
        mode: str = 'S0',
        seed: int = 42,
        timesteps: int = 1000,
        iterations: int = 5,
    ):
        super().__init__()
        if mode not in ('S0', 'S1', 'S2'):
            raise ValueError('mode must be S0, S1 or S2')
        if iterations != 5:
            raise ValueError('The registered platform uses exactly five refinement steps')
        if not isinstance(timesteps, int) or timesteps < iterations:
            raise ValueError('timesteps must be an integer at least as large as iterations')
        self.mode = mode
        self.method = {'S0': 'direct', 'S1': 'iterative', 'S2': 'diffusion'}[mode]
        self.iterations = iterations
        self.timesteps = timesteps
        self.init_seed = int(seed)

        # All modules are initially on CPU. Seed only the CPU default generator:
        # torch.manual_seed also touches CUDA, beyond fork_rng(devices=[]).
        # Two separate domains both use the one registered training seed42.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.init_seed)
            self.rgb_encoder = _PyramidEncoder(3, self.rgb_widths)
            self.decoder = _HighResolutionDecoder(self.rgb_widths)
            self.head = nn.Sequential(
                _norm(self.rgb_widths[0]), nn.SiLU(),
                nn.Conv2d(self.rgb_widths[0], self.classes, 1),
            )

        if mode != 'S0':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.init_seed)
                self.time_mlp = nn.Sequential(
                    nn.Linear(self.time_encoding_dim, self.time_embedding_dim),
                    nn.SiLU(),
                    nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
                )
                self.state_encoder = _PyramidEncoder(
                    self.classes, self.state_widths, self.time_embedding_dim
                )
                self.state_projection = nn.ModuleList([
                    nn.Conv2d(state_c, rgb_c, 1, bias=False)
                    for state_c, rgb_c in zip(self.state_widths, self.rgb_widths)
                ])

        # Same rounded descending grid used by five-step GaussianDiffusion.sample.
        self.register_buffer(
            'stage_timesteps',
            torch.linspace(timesteps - 1, 0, iterations).round().long(),
        )

    def encode_rgb(self, image: Tensor) -> tuple[Tensor, ...]:
        """Compute RGB features once; do not detach them during training."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('RGB input must have shape [B, 3, H, W]')
        return self.rgb_encoder(image)

    def _time_embedding(self, t: Tensor, state: Tensor) -> Tensor:
        if t.ndim != 1 or t.shape[0] != state.shape[0]:
            raise ValueError('Time indices must have shape [B]')
        half = self.time_encoding_dim // 2
        frequency = torch.exp(
            -math.log(10000) * torch.arange(half, device=state.device)
            / max(half - 1, 1)
        )
        angles = t.to(device=state.device, dtype=torch.float32)[:, None] * frequency[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        return self.time_mlp(embedding.to(dtype=state.dtype))

    def forward_with_rgb(
        self,
        features: Sequence[Tensor],
        state: Tensor | None = None,
        t: Tensor | None = None,
    ) -> Tensor:
        """One decode from cached RGB features; output is raw four-channel scores."""
        if len(features) != len(self.rgb_widths):
            raise ValueError('RGB cache must contain all four feature scales')
        batch = features[0].shape[0]
        for feature, channels in zip(features, self.rgb_widths):
            if feature.ndim != 4 or feature.shape[:2] != (batch, channels):
                raise ValueError('RGB cache has invalid batch or channel dimensions')
        if self.mode == 'S0':
            if state is not None or t is not None:
                raise ValueError('S0 does not execute a state or time path')
            fused = features
        else:
            if state is None or t is None:
                raise ValueError('S1/S2 single-step decoding requires state and time')
            expected = (batch, self.classes, *features[0].shape[-2:])
            if state.shape != expected:
                raise ValueError(f'State must have shape {expected}')
            embedding = self._time_embedding(t, state)
            state_features = self.state_encoder(state, embedding)
            fused = tuple(
                rgb + projection(dynamic)
                for rgb, dynamic, projection in zip(features, state_features, self.state_projection)
            )
        return self.head(self.decoder(fused))

    def forward(
        self,
        image: Tensor,
        state: Tensor | None = None,
        t: Tensor | None = None,
    ) -> Tensor:
        features = self.encode_rgb(image)
        if self.mode == 'S1':
            if state is not None or t is not None:
                raise ValueError('S1 forward initializes and runs five steps; use forward_with_rgb for a single step')
            state = image.new_full((image.shape[0], self.classes, *image.shape[-2:]), -0.5)
            for stage in self.stage_timesteps:
                logits = self.forward_with_rgb(features, state, stage.expand(image.shape[0]))
                state = 2 * logits.softmax(dim=1) - 1
            return logits
        return self.forward_with_rgb(features, state, t)


@torch.no_grad()
def sample_cached(
    model: StaticSemanticModel,
    diffusion,
    image: Tensor,
    noise: Tensor,
    steps: int = 5,
) -> Tensor:
    """Exact legacy eta=0 DDIM update, caching only static RGB features.

    Returns the final clipped four-channel x0 scores, as does the legacy
    single-head sampler. Apply softmax after sampling for probability fusion.
    Initial noise is supplied by the caller and never regenerated here.
    """
    if model.mode != 'S2':
        raise ValueError('Cached DDIM sampling requires an S2 model')
    if model.timesteps != diffusion.timesteps:
        raise ValueError('Model and diffusion timestep counts must match')
    if not 1 <= steps <= diffusion.timesteps:
        raise ValueError('Sampling steps must lie between 1 and T')
    expected = (image.shape[0], model.classes, *image.shape[-2:])
    if noise.shape != expected or noise.device != image.device:
        raise ValueError('Noise must match the RGB batch, spatial shape and device')
    if diffusion.abar.device != image.device:
        raise ValueError('Diffusion schedule and input must be on the same device')
    features = model.encode_rgb(image)
    grid = torch.linspace(diffusion.timesteps - 1, 0, steps, device=image.device).round().long()
    xt = noise.clone()
    for i, t in enumerate(grid):
        output = model.forward_with_rgb(features, xt, t.expand(image.shape[0]))
        x0 = output.clamp(-1, 1)
        if i == len(grid) - 1:
            return x0
        a, ap = diffusion.abar[t], diffusion.abar[grid[i + 1]]
        epsilon = (xt - a.sqrt() * x0) / (1 - a).sqrt()
        xt = ap.sqrt() * x0 + (1 - ap).sqrt() * epsilon
    raise RuntimeError('Empty sample schedule')

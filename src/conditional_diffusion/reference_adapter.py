"""Frozen RGB backbone with an independently trained, hard-disabled reference residual."""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .condition_model import ReferenceFusionUNet
from .model import norm


class ReferenceResidualAdapter(nn.Module):
    def __init__(self, base_channels, width):
        super().__init__()
        self.input = nn.Conv2d(base_channels + 11, width, 3, padding=1)
        self.norm1 = norm(width)
        self.time = nn.Linear(base_channels * 4, width)
        self.hidden = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = norm(width)
        self.output = nn.Conv2d(width, 4, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, rgb, contrast, state, q, embedding):
        h = self.input(torch.cat([features, rgb, contrast, state, q], dim=1))
        h = F.silu(self.norm1(h) + self.time(F.silu(embedding))[:, :, None, None])
        return self.output(F.silu(self.norm2(self.hidden(h))))


class FrozenReferenceAdapter(nn.Module):
    method = 'diffusion'
    iterations = 5

    def __init__(self, config):
        super().__init__()
        path = Path(config['adapter_base_checkpoint'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != config['adapter_base_sha256']:
            raise ValueError('Frozen F0 base checkpoint hash mismatch')
        state = torch.load(path, map_location='cpu', weights_only=False)
        c = state['config']
        if not (c['method'] == 'diffusion' and c.get('condition_architecture') == 'split_rgb_state'
                and c.get('condition_fusion') == 'mean' and c.get('reference_enabled') is False
                and c['base_channels'] == config['base_channels']):
            raise ValueError('Adapter requires matching no-reference F0 architecture')
        self.base = ReferenceFusionUNet(c['base_channels'], 'mean')
        self.base.load_state_dict(state['model'], strict=True)
        self.base.requires_grad_(False)
        self.base.eval()
        self.reference_enabled = config['adapter_reference_enabled']
        self.adapter = ReferenceResidualAdapter(c['base_channels'], config['adapter_width'])
        self._features = None
        self.base.out.register_forward_pre_hook(self._capture_features)

    def _capture_features(self, module, args):
        self._features = args[0]

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, image, state=None, t=None):
        if image.shape[1] != 7 or state is None or t is None:
            raise ValueError('Adapter expects RGB, unweighted C, image-level q, state and t')
        q = image[:, 6:7]
        if not torch.equal(q, q[:, :, :1, :1].expand_as(q)):
            raise ValueError('Reference confidence must be constant over the image/patch')
        active = q[:, 0, 0, 0] > 0
        rgb_only = torch.cat([image[:, :3], torch.zeros_like(image[:, 3:])], dim=1)
        with torch.no_grad():
            baseline = self.base(rgb_only, state, t)
            features = self._features
            self._features = None
        # Exact bypass: no adapter operations or altered base batching for q=0.
        if not active.any():
            return baseline
        with torch.no_grad():
            half = self.base.embed_dim // 2
            freq = torch.exp(-math.log(10000) * torch.arange(half, device=image.device) / max(half - 1, 1))
            angles = t[active].float()[:, None] * freq[None]
            embedding = self.base.time_mlp(torch.cat([angles.sin(), angles.cos()], dim=1))
        contrast = image[active, 3:6] if self.reference_enabled else torch.zeros_like(image[active, 3:6])
        residual = self.adapter(features[active], image[active, :3], contrast, state[active], q[active], embedding)
        result = baseline.clone()
        result[active] = baseline[active] + q[active] * residual
        return result

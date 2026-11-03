"""Opt-in split RGB/state conditioning. Common architecture for F0/F1 controls."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .model import SegUNet, norm


class ReferenceFusionUNet(SegUNet):
    def __init__(self, base=32, fusion='mean'):
        if fusion not in ['mean', 'adaptive']:
            raise ValueError('Unknown condition fusion')
        super().__init__('diffusion', base, 5, False)
        del self.input
        self.fusion = fusion
        widths = [base * m for m in [1, 2, 4, 8]]
        self.state_input = nn.Conv2d(4, base, 3, padding=1)
        # Both F0 and F1 have seven image feature channels: RGB, q*C, q.
        # F0 supplies exactly zero for the last four. Parameter/RNG layouts match.
        self.image_input = nn.Conv2d(7, base, 3, padding=1)
        self.image_down = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1)
                                        for a, b in zip(widths, widths[1:])])
        self.image_norm = nn.ModuleList([norm(w) for w in widths])
        if fusion == 'adaptive':
            self.gates = nn.ModuleList([nn.Conv2d(2 * w + 1, w, 1) for w in widths])
            self.gate_time = nn.ModuleList([nn.Linear(base * 4, w) for w in widths])

    def backbone(self, image, state=None, t=None):
        if image.shape[1] != 7 or state is None or state.shape[1] != 4 or t is None:
            raise ValueError('Split conditioning requires RGB+four auxiliary features, noisy four-class state and timestep')
        half = self.embed_dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=image.device) / max(half - 1, 1))
        angles = t.float()[:, None] * freq[None]
        emb = self.time_mlp(torch.cat([angles.sin(), angles.cos()], dim=1))
        h, condition = self.state_input(state), self.image_input(image)
        skips = []
        for i, blocks in enumerate(self.enc):
            evidence = F.silu(self.image_norm[i](condition))
            if self.fusion == 'mean':
                h = .5 * (h + evidence)
            else:
                q = F.interpolate(image[:, 6:7], size=h.shape[-2:], mode='nearest')
                gate = torch.sigmoid(self.gates[i](torch.cat([h, evidence, q], dim=1))
                                     + self.gate_time[i](F.silu(emb))[:, :, None, None])
                h = (1 - gate) * h + gate * evidence
            for block in blocks:
                h = block(h, emb)
            skips.append(h)
            if i < len(self.down):
                h, condition = self.down[i](h), self.image_down[i](condition)
        for up, blocks, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            h = up(F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False))
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, emb)
        return self.out(h)

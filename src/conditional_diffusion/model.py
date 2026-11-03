import math
import copy
import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, time_dim):
        super().__init__()
        self.first = nn.Sequential(norm(cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1))
        self.second = nn.Sequential(norm(cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.time = nn.Linear(time_dim, cout) if time_dim else None

    def forward(self, x, time):
        h = self.first(x)
        if self.time is not None:
            h = h + self.time(F.silu(time))[:, :, None, None]
        return self.skip(x) + self.second(h)


class SegUNet(nn.Module):
    def __init__(self, method='diffusion', base=32, iterations=5, dual_head=False):
        super().__init__()
        if method not in ['direct', 'iterative', 'diffusion']:
            raise ValueError(method)
        self.method, self.iterations = method, iterations
        if dual_head and method != 'diffusion':
            raise ValueError('Dual heads require diffusion')
        self.dual_head = dual_head
        widths = [base*m for m in [1, 2, 4, 8]]
        self.embed_dim = base
        td = base*4 if method == 'diffusion' else 0
        self.time_mlp = nn.Sequential(nn.Linear(base, td), nn.SiLU(), nn.Linear(td, td)) if td else None
        self.input = nn.Conv2d(3 if method == 'direct' else 7, base, 3, padding=1)
        self.enc = nn.ModuleList([nn.ModuleList([ResidualBlock(w, w, td), ResidualBlock(w, w, td)]) for w in widths])
        self.down = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(widths, widths[1:])])
        self.up = nn.ModuleList([nn.Conv2d(widths[i+1], widths[i], 3, padding=1) for i in [2, 1, 0]])
        self.dec = nn.ModuleList([nn.ModuleList([ResidualBlock(2*widths[i], widths[i], td), ResidualBlock(widths[i], widths[i], td)]) for i in [2, 1, 0]])
        self.out = nn.Sequential(norm(base), nn.SiLU(), nn.Conv2d(base, 4, 1))
        # Independent parameters, identical initial outputs, no extra RNG draws.
        if dual_head:
            self.denoise_out = copy.deepcopy(self.out)

    def backbone(self, image, state=None, t=None):
        emb = None
        if self.time_mlp is not None:
            half = self.embed_dim//2
            freq = torch.exp(-math.log(10000)*torch.arange(half, device=image.device)/(max(half-1, 1)))
            angles = t.float()[:, None]*freq[None]
            emb = self.time_mlp(torch.cat([angles.sin(), angles.cos()], dim=1))
        h = self.input(image if state is None else torch.cat([image, state], dim=1))
        skips = []
        for i, blocks in enumerate(self.enc):
            for block in blocks:
                h = block(h, emb)
            skips.append(h)
            if i < len(self.down):
                h = self.down[i](h)
        for up, blocks, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            h = up(F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False))
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, emb)
        logits = self.out(h)
        if self.dual_head:
            return {'logits': logits, 'x0': self.denoise_out(h)}
        return logits

    def forward(self, image, state=None, t=None):
        if self.method == 'iterative':
            state = image.new_full((image.shape[0], 4, *image.shape[-2:]), .25)
            for _ in range(self.iterations):
                logits = self.backbone(image, state)
                state = logits.softmax(dim=1)
            return logits
        return self.backbone(image, state, t)


def make_model(config):
    if config.get('condition_architecture') == 'frozen_reference_adapter':
        from .reference_adapter import FrozenReferenceAdapter
        return FrozenReferenceAdapter(config)
    if config.get('condition_architecture') == 'split_rgb_state':
        from .condition_model import ReferenceFusionUNet
        if config['method'] != 'diffusion' or config.get('dual_head', False):
            raise ValueError('Split conditioning requires single-head diffusion')
        return ReferenceFusionUNet(config['base_channels'], config['condition_fusion'])
    return SegUNet(config['method'], config['base_channels'], config['iterations'], config.get('dual_head', False))

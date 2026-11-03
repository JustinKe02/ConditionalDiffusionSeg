"""RGB-conditioned mask denoiser with the existing GLP global/local backbone.

Network blocks are adapted from work/glpflow_core.py (2026-10-21, lines
170-419). Historical experiment code is not imported. This module changes the
state input and time interface, while retaining the multiscale RGB encoder,
global prior head, neighborhood attention, and three-stream joint attention.
The diffusion process and target-state encoding belong to the caller.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint


def _positions(height, width, channels, box=None, device="cpu"):
    y0, x0, y1, x1 = (0.0, 0.0, 1.0, 1.0) if box is None else box
    ys = y0 + (torch.arange(height, device=device) + 0.5) / height * (y1 - y0)
    xs = x0 + (torch.arange(width, device=device) + 0.5) / width * (x1 - x0)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    frequency = 2 ** torch.arange(channels // 4, device=device, dtype=torch.float32) * math.pi
    return torch.cat([
        function(value.reshape(-1, 1) * frequency)
        for value in (yy, xx) for function in (torch.sin, torch.cos)
    ], dim=1)


def _attention(query, key, value, valid=None):
    scores = (query * query.shape[-1] ** -0.5) @ key.transpose(-2, -1)
    if valid is not None:
        scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
    return scores.softmax(-1) @ value


def _run(block, *args):
    if torch.is_grad_enabled() and block.training:
        return checkpoint(block, *args, use_reentrant=False)
    return block(*args)


class _ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, value):
        return value * self.projection(value.mean((2, 3), keepdim=True)).sigmoid()


class _Modulation(nn.Module):
    def __init__(self, channels, condition_dim, spatial=True):
        super().__init__()
        self.vector = nn.Linear(condition_dim, 6 * channels)
        self.spatial = nn.Conv2d(1, 6 * channels, 1) if spatial else None
        for layer in (self.vector, self.spatial):
            if layer is not None:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, condition, uncertainty, size):
        modulation = self.vector(F.silu(condition))[:, :, None, None]
        if self.spatial is not None:
            modulation = modulation + self.spatial(F.adaptive_avg_pool2d(uncertainty, size))
        return modulation.chunk(6, dim=1)


class _ConvBlock(nn.Module):
    def __init__(self, channels, condition_dim=0, spatial=True):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels, affine=not condition_dim)
        self.norm2 = nn.GroupNorm(8, channels, affine=not condition_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(), _ChannelAttention(channels), nn.Conv2d(channels, channels, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 1),
            nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels),
            nn.GELU(), nn.Conv2d(2 * channels, channels, 1),
        )
        self.modulation = _Modulation(channels, condition_dim, spatial) if condition_dim else None

    def forward(self, value, condition=None, uncertainty=None):
        if self.modulation is None:
            value = value + self.conv(self.norm1(value))
            return value + self.ffn(self.norm2(value))
        scale1, bias1, gate1, scale2, bias2, gate2 = self.modulation(condition, uncertainty, value.shape[-2:])
        value = value + gate1 * self.conv(self.norm1(value) * (1 + scale1) + bias1)
        return value + gate2 * self.ffn(self.norm2(value) * (1 + scale2) + bias2)


class _NeighborhoodBlock(nn.Module):
    def __init__(self, channels, condition_dim, heads, kernel):
        super().__init__()
        self.heads, self.kernel = heads, kernel
        self.norm1 = nn.GroupNorm(8, channels, affine=False)
        self.norm2 = nn.GroupNorm(8, channels, affine=False)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.projection = nn.Conv2d(channels, channels, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 1),
            nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels),
            nn.GELU(), nn.Conv2d(2 * channels, channels, 1),
        )
        self.relative = nn.Parameter(torch.zeros(heads, kernel * kernel, 1))
        self.modulation = _Modulation(channels, condition_dim)

    def attend(self, value):
        batch, channels, height, width = value.shape
        depth, kernel = channels // self.heads, self.kernel
        query, key, values = self.qkv(value).chunk(3, dim=1)
        query = query.reshape(batch, self.heads, depth, 1, height * width) * depth ** -0.5
        key = F.unfold(key, kernel, padding=kernel // 2).reshape(batch, self.heads, depth, kernel * kernel, height * width)
        values = F.unfold(values, kernel, padding=kernel // 2).reshape(batch, self.heads, depth, kernel * kernel, height * width)
        valid = F.unfold(value.new_ones(1, 1, height, width), kernel, padding=kernel // 2)
        valid = valid.reshape(1, 1, kernel * kernel, height * width) > 0
        scores = (query * key).sum(2) + self.relative
        weights = scores.masked_fill(~valid, torch.finfo(scores.dtype).min).softmax(2)
        attended = (weights[:, :, None] * values).sum(3).reshape(batch, channels, height, width)
        return self.projection(attended)

    def forward(self, value, condition, uncertainty):
        scale1, bias1, gate1, scale2, bias2, gate2 = self.modulation(condition, uncertainty, value.shape[-2:])
        value = value + gate1 * self.attend(self.norm1(value) * (1 + scale1) + bias1)
        return value + gate2 * self.ffn(self.norm2(value) * (1 + scale2) + bias2)


class _TransformerBlock(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.projection = nn.Linear(channels, channels)
        self.mlp = nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(), nn.Linear(4 * channels, channels))

    def forward(self, value):
        batch, length, channels = value.shape
        projected = self.qkv(self.norm1(value))
        query, key, values = projected.reshape(
            batch, length, 3, self.heads, channels // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = _attention(query, key, values).transpose(1, 2).reshape(batch, length, channels)
        value = value + self.projection(attended)
        return value + self.mlp(self.norm2(value))


class _JointBlock(nn.Module):
    def __init__(self, channels, condition_dim, heads):
        super().__init__()
        self.heads = heads
        self.streams = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(channels, elementwise_affine=False),
                "norm2": nn.LayerNorm(channels, elementwise_affine=False),
                "qkv": nn.Linear(channels, 3 * channels),
                "projection": nn.Linear(channels, channels),
                "mlp": nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(), nn.Linear(4 * channels, channels)),
                "modulation": nn.Linear(condition_dim, 6 * channels),
            }) for _ in range(3)
        ])
        for stream in self.streams:
            nn.init.zeros_(stream["modulation"].weight)
            nn.init.zeros_(stream["modulation"].bias)

    def forward(self, tokens, condition, valid):
        modulations, projections = [], []
        lengths = [value.shape[1] for value in tokens]
        for value, stream in zip(tokens, self.streams):
            modulation = stream["modulation"](F.silu(condition))[:, None].chunk(6, dim=2)
            modulations.append(modulation)
            projections.append(stream["qkv"](stream["norm1"](value) * (1 + modulation[0]) + modulation[1]))
        batch, _, channels = tokens[0].shape
        query, key, values = torch.cat(projections, dim=1).reshape(
            batch, sum(lengths), 3, self.heads, channels // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = _attention(query, key, values, valid).transpose(1, 2).reshape(batch, sum(lengths), channels)
        outputs = []
        for value, update, modulation, stream in zip(tokens, attended.split(lengths, dim=1), modulations, self.streams):
            value = value + modulation[2] * stream["projection"](update)
            outputs.append(value + modulation[5] * stream["mlp"](stream["norm2"](value) * (1 + modulation[3]) + modulation[4]))
        return outputs


class _GlobalEncoder(nn.Module):
    def __init__(self, widths, token_dim, heads):
        super().__init__()
        first, second, third, _ = widths
        self.stem = nn.Sequential(nn.Conv2d(3, first, 3, 2, 1), _ConvBlock(first))
        self.level1 = nn.Sequential(nn.Conv2d(first, second, 3, 2, 1), _ConvBlock(second), _ConvBlock(second))
        self.level2 = nn.Sequential(nn.Conv2d(second, third, 3, 2, 1), _ConvBlock(third), _ConvBlock(third))
        self.down3, self.down4 = nn.Conv2d(third, token_dim, 3, 2, 1), nn.Conv2d(token_dim, token_dim, 3, 2, 1)
        self.blocks3 = nn.ModuleList([_TransformerBlock(token_dim, heads) for _ in range(2)])
        self.blocks4 = nn.ModuleList([_TransformerBlock(token_dim, heads) for _ in range(2)])
        self.lateral = nn.ModuleList([nn.Conv2d(channels, third, 1) for channels in (second, third, token_dim, token_dim)])
        self.prior_head = nn.Sequential(_ConvBlock(third), nn.GroupNorm(8, third), nn.GELU(), nn.Conv2d(third, 4, 1))
        self.token_projection = nn.Linear(token_dim, token_dim)

    @staticmethod
    def sequence(value, blocks):
        batch, channels, height, width = value.shape
        tokens = value.flatten(2).transpose(1, 2) + _positions(height, width, channels, device=value.device)[None]
        for block in blocks:
            tokens = block(tokens)
        return tokens, tokens.transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, thumbnail):
        first = self.level1(self.stem(thumbnail))
        second = self.level2(first)
        _, third = self.sequence(self.down3(second), self.blocks3)
        tokens, fourth = self.sequence(self.down4(third), self.blocks4)
        top = None
        for value, projection in zip((fourth, third, second, first), reversed(self.lateral)):
            lateral = projection(value)
            top = lateral if top is None else lateral + F.interpolate(top, size=lateral.shape[-2:], mode="nearest")
        return self.token_projection(tokens), self.prior_head(top), fourth.shape[-2:]


class _LocalEncoder(nn.Module):
    def __init__(self, widths):
        super().__init__()
        channels = (3, *widths)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[index], width, 3, stride=1 if index == 0 else 2, padding=1),
                *[_ConvBlock(width) for _ in range(1 if index < 2 else 2)],
            ) for index, width in enumerate(widths)
        ])

    def forward(self, image):
        features = []
        for layer in self.layers:
            image = layer[0](image)
            for block in layer[1:]:
                image = _run(block, image)
            features.append(image)
        return features


class ConditionalMaskDenoiser(nn.Module):
    """Predict four raw class logits from a noisy mask and cached RGB context.

    Boxes use normalized full-image coordinates in (y0, x0, y1, x1) order.
    Arbitrary positive window sizes are padded on the bottom/right to a multiple
    of 16. Denoising outputs are cropped back to the original window shape.
    Cached priors and entropy are detached. Cached RGB/global tokens retain
    gradients so a training loss can update both image encoders.
    """

    def __init__(self, state_channels=3, timesteps=1000, widths=(32, 64, 128, 192), token_dim=256, heads=4, na_kernel=7):
        super().__init__()
        if state_channels not in (3, 4) or not isinstance(timesteps, int) or timesteps < 2:
            raise ValueError("Expected 3/4 state channels and at least two timesteps")
        if len(widths) != 4 or any(width < 8 or width % 8 for width in widths):
            raise ValueError("Four channel widths must be positive multiples of eight")
        if heads < 1 or token_dim < 8 or token_dim % 8 or token_dim % heads or widths[-1] % heads:
            raise ValueError("Attention channels must be divisible by heads and token_dim by eight")
        if na_kernel < 1 or na_kernel % 2 != 1:
            raise ValueError("Neighborhood kernel must be a positive odd integer")
        self.state_channels, self.timesteps = state_channels, timesteps
        self.local = _LocalEncoder(widths)
        self.global_encoder = _GlobalEncoder(widths, token_dim, heads)
        self.time_mlp = nn.Sequential(nn.Linear(128, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.state_stem = nn.Conv2d(state_channels + 5, widths[0], 3, padding=1)
        self.down = nn.ModuleList([nn.Conv2d(first, second, 3, 2, 1) for first, second in zip(widths, widths[1:])])
        self.encoder_blocks = nn.ModuleList([
            nn.ModuleList([_ConvBlock(widths[0], token_dim, spatial=False)]),
            nn.ModuleList([_ConvBlock(widths[1], token_dim, spatial=False)]),
            nn.ModuleList([_ConvBlock(widths[2], token_dim) for _ in range(2)]),
            nn.ModuleList([_NeighborhoodBlock(widths[3], token_dim, heads, na_kernel) for _ in range(2)]),
        ])
        self.rgb_fusion = nn.ModuleList([nn.Conv2d(width, width, 1) for width in widths])
        self.state_token = nn.Conv2d(widths[-1], token_dim, 3, 2, 1)
        self.rgb_token = nn.Conv2d(widths[-1], token_dim, 3, 2, 1)
        self.joint = nn.ModuleList([_JointBlock(token_dim, token_dim, heads) for _ in range(2)])
        decoder_channels = (token_dim, *reversed(widths))
        self.up = nn.ModuleList([
            nn.Conv2d(first, second, 3, padding=1)
            for first, second in zip(decoder_channels, decoder_channels[1:])
        ])
        self.decoder_blocks = nn.ModuleList([
            _NeighborhoodBlock(widths[3], token_dim, heads, na_kernel),
            _ConvBlock(widths[2], token_dim),
            _ConvBlock(widths[1], token_dim, spatial=False),
            _ConvBlock(widths[0], token_dim, spatial=False),
        ])
        self.head = nn.Sequential(nn.GroupNorm(8, widths[0]), nn.SiLU(), nn.Conv2d(widths[0], 4, 1))

    def encode_global(self, thumbnails: Sequence[Tensor]):
        contexts = []
        for thumbnail in thumbnails:
            if thumbnail.ndim != 3 or thumbnail.shape[0] != 3 or min(thumbnail.shape[-2:]) < 32:
                raise ValueError("Each RGB thumbnail must have shape [3,H,W] with H,W >= 32")
            tokens, prior, grid = self.global_encoder(thumbnail[None])
            contexts.append((tokens[0], prior[0], grid))
        return contexts

    @staticmethod
    def _window_prior(prior_logits, box, height, width):
        y0, x0, y1, x1 = box
        ys = y0 + (torch.arange(height, device=prior_logits.device) + 0.5) / height * (y1 - y0)
        xs = x0 + (torch.arange(width, device=prior_logits.device) + 0.5) / width * (x1 - x0)
        yy, xx = torch.meshgrid(ys * 2 - 1, xs * 2 - 1, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)[None]
        return F.grid_sample(prior_logits[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]

    def encode_condition(self, image: Tensor, boxes: Tensor, global_context):
        if image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 3 or min(image.shape[-2:]) < 1:
            raise ValueError("RGB input must have shape [B,3,H,W] with positive B,H,W")
        batch, _, height, width = image.shape
        boxes = torch.as_tensor(boxes, device=image.device, dtype=image.dtype)
        if (boxes.shape != (batch, 4) or not bool(torch.isfinite(boxes).all())
                or not bool((boxes[:, 2:] > boxes[:, :2]).all())):
            raise ValueError("Boxes must have shape [B,4] and finite positive extents")
        if len(global_context) != batch:
            raise ValueError("Each window requires one matching global context")
        pad = (0, (-width) % 16, 0, (-height) % 16)
        padded = F.pad(image, pad, mode="replicate")
        with torch.no_grad():
            prior_logits = torch.stack([
                self._window_prior(context[1].detach(), box, height, width)
                for context, box in zip(global_context, boxes)
            ])
            prior = F.pad(prior_logits.softmax(1), pad, mode="replicate")
            safe_prior = prior.clamp_min(1e-6)
            uncertainty = (-(safe_prior * safe_prior.log()).sum(1, keepdim=True) / math.log(4)).clamp(0, 1)
        tokens = pad_sequence([context[0] for context in global_context], batch_first=True)
        positions = pad_sequence([
            _positions(*context[2], tokens.shape[-1], device=image.device)
            for context in global_context
        ], batch_first=True)
        lengths = torch.tensor([context[0].shape[0] for context in global_context], device=image.device)
        valid = torch.arange(tokens.shape[1], device=image.device)[None] < lengths[:, None]
        padded_boxes = boxes.clone()
        scale = boxes.new_tensor([padded.shape[-2] / height, padded.shape[-1] / width])
        padded_boxes[:, 2:] = boxes[:, :2] + (boxes[:, 2:] - boxes[:, :2]) * scale
        return {"features": self.local(padded), "prior": prior, "uncertainty": uncertainty,
                "tokens": tokens, "positions": positions, "valid": valid, "boxes": padded_boxes,
                "shape": (height, width), "pad": pad}

    def denoise(self, condition, noisy_state: Tensor, timesteps: Tensor):
        features = condition["features"]
        batch = features[0].shape[0]
        expected = (batch, self.state_channels, *condition["shape"])
        if noisy_state.shape != expected or noisy_state.device != features[0].device:
            raise ValueError(f"Noisy state must have shape {expected} and match the condition device")
        if timesteps.shape != (batch,) or timesteps.dtype not in (torch.int32, torch.int64):
            raise ValueError("Timesteps must be integer indices with shape [B]")
        if not bool(((timesteps >= 0) & (timesteps < self.timesteps)).all()):
            raise ValueError("Time index outside the diffusion schedule")
        frequency = torch.exp(-math.log(10000) * torch.arange(64, device=noisy_state.device) / 64)
        angles = timesteps.to(device=noisy_state.device, dtype=torch.float32)[:, None] * frequency[None]
        time = self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=1).to(noisy_state.dtype))
        state = F.pad(noisy_state, condition["pad"], mode="replicate")
        uncertainty = condition["uncertainty"]
        value = self.state_stem(torch.cat((state, condition["prior"], uncertainty), dim=1))
        skips = []
        for index, blocks in enumerate(self.encoder_blocks):
            if index:
                value = self.down[index - 1](value)
            value = value + self.rgb_fusion[index](features[index])
            for block in blocks:
                value = _run(block, value, time, uncertainty)
            skips.append(value)
        state_tokens, rgb_tokens = self.state_token(value), self.rgb_token(features[-1])
        _, channels, height, width = state_tokens.shape
        positions = torch.stack([_positions(height, width, channels, box, noisy_state.device) for box in condition["boxes"]])
        streams = [state_tokens.flatten(2).transpose(1, 2) + positions,
                   rgb_tokens.flatten(2).transpose(1, 2) + positions,
                   condition["tokens"] + condition["positions"]]
        local_valid = torch.ones(batch, 2 * height * width, dtype=torch.bool, device=noisy_state.device)
        valid = torch.cat((local_valid, condition["valid"]), dim=1)
        for block in self.joint:
            streams = _run(block, streams, time, valid)
        value = streams[0].transpose(1, 2).reshape(batch, channels, height, width)
        for up, block, skip in zip(self.up, self.decoder_blocks, reversed(skips)):
            value = up(F.interpolate(value, size=skip.shape[-2:], mode="nearest"))
            value = (value + skip) / math.sqrt(2)
            value = _run(block, value, time, uncertainty)
        logits = self.head(value)
        original_height, original_width = condition["shape"]
        return logits[:, :, :original_height, :original_width]

    def forward(self, image, noisy_state, timesteps, boxes, global_context):
        return self.denoise(self.encode_condition(image, boxes, global_context), noisy_state, timesteps)

"""Pure-PyTorch baseline segmentation architectures and deterministic operators.

Implements:
  - Deterministic bilinear interpolation (det_interpolate: Y = A_h X A_w^T)
  - Deterministic adaptive average pooling (det_adaptive_avg_pool2d: Y = P_h X P_w^T)
  - Deterministic 2D bilinear grid sampling (det_grid_sample via torch.gather)
  - U-Net (Ronneberger et al., MICCAI 2015)
  - UNet++ (Zhou et al., IEEE TMI 2020)
  - PSPNet (Zhao et al., CVPR 2017)
  - DeepLabv3+ (Chen et al., ECCV 2018)
  - HRNetV2-W48 (Wang et al., TPAMI 2021)
  - SegFormer-B1 (Xie et al., NeurIPS 2021)
  - Mask2Former (Cheng et al., CVPR 2022)
"""
import itertools
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


# -----------------------------------------------------------------------------
# Deterministic operators (exact forward & deterministic backward on CUDA)
# -----------------------------------------------------------------------------

_INTERP_CACHE = {}
_POOL_CACHE = {}


def _bilinear_matrix(n_in: int, n_out: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n_in, n_out, device, dtype)
    mat = _INTERP_CACHE.get(key)
    if mat is not None:
        return mat
    if n_in == n_out:
        mat = torch.eye(n_in, device=device, dtype=dtype)
    else:
        dst = torch.arange(n_out, device=device, dtype=torch.float64)
        src = (dst + 0.5) * (n_in / n_out) - 0.5
        src_clamped = src.clamp(0.0, float(n_in - 1))
        i0 = torch.floor(src_clamped).long()
        i1 = (i0 + 1).clamp(max=n_in - 1)
        w1 = (src_clamped - i0.to(torch.float64)).to(dtype)
        w0 = (1.0 - w1).to(dtype)
        mat = torch.zeros((n_out, n_in), device=device, dtype=dtype)
        rows = torch.arange(n_out, device=device)
        mat[rows, i0] += w0
        mat[rows, i1] += w1
    _INTERP_CACHE[key] = mat
    return mat


def det_interpolate(x: torch.Tensor, size=None, scale_factor=None) -> torch.Tensor:
    """Deterministic bilinear interpolation matching F.interpolate(..., mode='bilinear', align_corners=False)."""
    h_in, w_in = x.shape[-2], x.shape[-1]
    if size is not None:
        if isinstance(size, int):
            h_out = w_out = size
        else:
            h_out, w_out = int(size[0]), int(size[1])
    elif scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            h_out, w_out = int(round(h_in * scale_factor)), int(round(w_in * scale_factor))
        else:
            h_out, w_out = int(round(h_in * scale_factor[0])), int(round(w_in * scale_factor[1]))
    else:
        raise ValueError("Either size or scale_factor must be specified")
    if h_out == h_in and w_out == w_in:
        return x
    ah = _bilinear_matrix(h_in, h_out, x.device, x.dtype)
    aw = _bilinear_matrix(w_in, w_out, x.device, x.dtype)
    return torch.matmul(ah, torch.matmul(x, aw.t()))


def _pool_matrix(n_in: int, n_out: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n_in, n_out, device, dtype)
    mat = _POOL_CACHE.get(key)
    if mat is not None:
        return mat
    mat = torch.zeros((n_out, n_in), device=device, dtype=dtype)
    for i in range(n_out):
        start = (i * n_in) // n_out
        end = ((i + 1) * n_in + n_out - 1) // n_out
        mat[i, start:end] = 1.0 / float(end - start)
    _POOL_CACHE[key] = mat
    return mat


def det_adaptive_avg_pool2d(x: torch.Tensor, output_size) -> torch.Tensor:
    """Deterministic adaptive avg pool2d matching F.adaptive_avg_pool2d."""
    if isinstance(output_size, int):
        h_out = w_out = output_size
    else:
        h_out, w_out = int(output_size[0]), int(output_size[1])
    if h_out == 1 and w_out == 1:
        return x.mean(dim=(-2, -1), keepdim=True)
    h_in, w_in = x.shape[-2], x.shape[-1]
    if h_out == h_in and w_out == w_in:
        return x
    ph = _pool_matrix(h_in, h_out, x.device, x.dtype)
    pw = _pool_matrix(w_in, w_out, x.device, x.dtype)
    return torch.matmul(ph, torch.matmul(x, pw.t()))


def det_grid_sample(v: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Deterministic bilinear grid_sample matching F.grid_sample(v, grid, mode='bilinear', padding_mode='zeros', align_corners=False).
    v: (N, C, H, W), grid: (N, Ho, Wo, 2) with (x, y) in [-1, 1].
    """
    N, C, H, W = v.shape
    _, Ho, Wo, _ = grid.shape
    K = Ho * Wo
    gx = grid[..., 0].reshape(N, K)
    gy = grid[..., 1].reshape(N, K)
    x = ((gx + 1.0) * W - 1.0) * 0.5
    y = ((gy + 1.0) * H - 1.0) * 0.5
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (x - x0.to(x.dtype)).unsqueeze(1)
    wy = (y - y0.to(y.dtype)).unsqueeze(1)
    v_flat = v.reshape(N, C, H * W)

    def sample_corner(yi, xi):
        valid = ((yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)).unsqueeze(1).to(v.dtype)
        idx = (yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)).unsqueeze(1).expand(N, C, K)
        return torch.gather(v_flat, 2, idx) * valid

    v00 = sample_corner(y0, x0)
    v01 = sample_corner(y0, x1)
    v10 = sample_corner(y1, x0)
    v11 = sample_corner(y1, x1)
    out = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)
    return out.reshape(N, C, Ho, Wo)


def det_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
    """Deterministic multi-head attention: q, k, v of shape (B, H, Nq, D) / (B, H, Nk, D)."""
    scale = q.shape[-1] ** -0.5
    logits = torch.matmul(q * scale, k.transpose(-2, -1))
    if attn_mask is not None:
        logits = logits.masked_fill(attn_mask, -1e4)
    attn = logits.softmax(dim=-1)
    return torch.matmul(attn, v)


def run_ckpt(module: nn.Module, *args):
    """Activation recomputation during training only; numerically identical forward."""
    if torch.is_grad_enabled() and module.training:
        return checkpoint(module, *args, use_reentrant=False)
    return module(*args)


def fg_dice_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Foreground 3-class mean Dice loss (identical to glpflow_core.dice_loss)."""
    probs = logits.softmax(1)
    gt = F.one_hot(mask, 4).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * gt).sum(dims)
    denom = probs.sum(dims) + gt.sum(dims)
    return 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6))[1:].mean()


def standard_sem_loss(logits: torch.Tensor, mask: torch.Tensor):
    ce = F.cross_entropy(logits, mask, reduction='none').mean()
    dice = fg_dice_loss(logits, mask)
    total = 0.5 * ce + 0.5 * dice
    return total, {'loss_total': total, 'loss_ce': ce, 'loss_dice_fg': dice}


# -----------------------------------------------------------------------------
# 1. U-Net (Ronneberger et al., MICCAI 2015) - 5 levels, 64..1024, BN, ~31.04M
# -----------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, num_classes: int = 4, channels=(64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc1 = DoubleConv(3, channels[0])
        self.enc2 = DoubleConv(channels[0], channels[1])
        self.enc3 = DoubleConv(channels[1], channels[2])
        self.enc4 = DoubleConv(channels[2], channels[3])
        self.bot = DoubleConv(channels[3], channels[4])
        self.pool = nn.MaxPool2d(2, 2)

        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(channels[4], channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(channels[3], channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(channels[2], channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(channels[1], channels[0])
        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bot(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

    def forward_train(self, x: torch.Tensor, mask: torch.Tensor):
        logits = self.forward(x)
        _, terms = standard_sem_loss(logits, mask)
        return logits, terms


# -----------------------------------------------------------------------------
# 2. UNet++ (Zhou et al., IEEE TMI 2020) - VGG blocks 32..512, nested dense skips, ~9.16M
# -----------------------------------------------------------------------------

class VGGBlock(nn.Module):
    def __init__(self, c_in: int, c_mid: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes: int = 4, nb_filter=(32, 64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = VGGBlock(3, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        return det_interpolate(x, scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3)], 1))
        return self.final(x0_4)

    def forward_train(self, x: torch.Tensor, mask: torch.Tensor):
        logits = self.forward(x)
        _, terms = standard_sem_loss(logits, mask)
        return logits, terms


# -----------------------------------------------------------------------------
# Shared ResNet-50 Backbone (for PSPNet, DeepLabv3+, Mask2Former, DDP)
# -----------------------------------------------------------------------------

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet50Backbone(nn.Module):
    def __init__(self, replace_stride_with_dilation=(False, False, False), deep_stem: bool = False):
        super().__init__()
        self.inplanes = 128 if deep_stem else 64
        if deep_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dilation = 1
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(256, 6, stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, 3, stride=2, dilate=replace_stride_with_dilation[2])

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, dilate: bool = False):
        downsample = None
        prev_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [Bottleneck(self.inplanes, planes, stride=stride, dilation=prev_dilation, downsample=downsample)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, stride=1, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.maxpool(self.stem(x))
        c1 = self.layer1(x)   # 256, 1/4
        c2 = self.layer2(c1)  # 512, 1/8
        c3 = self.layer3(c2)  # 1024, 1/8 or 1/16
        c4 = self.layer4(c3)  # 2048, 1/8, 1/16 or 1/32
        return c1, c2, c3, c4


# -----------------------------------------------------------------------------
# 3. PSPNet (Zhao et al., CVPR 2017) - ResNet-50 OS=8, PPM (1,2,3,6), aux 0.4, ~49.07M
# -----------------------------------------------------------------------------

class PSPNet(nn.Module):
    def __init__(self, num_classes: int = 4, bins=(1, 2, 3, 6), aux_weight: float = 0.4):
        super().__init__()
        self.aux_weight = aux_weight
        self.backbone = ResNet50Backbone(replace_stride_with_dilation=(False, True, True), deep_stem=True)
        self.bins = bins
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 512, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            for _ in bins
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048 + 512 * len(bins), 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1),
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def _decode(self, c3: torch.Tensor, c4: torch.Tensor, hw):
        h, w = c4.shape[-2:]
        priors = [c4]
        for b, stage in zip(self.bins, self.ppm):
            pooled = det_adaptive_avg_pool2d(c4, (b, b))
            priors.append(det_interpolate(stage(pooled), size=(h, w)))
        main_logits = det_interpolate(self.bottleneck(torch.cat(priors, dim=1)), size=hw)
        return main_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, c3, c4 = self.backbone(x)
        return self._decode(c3, c4, x.shape[-2:])

    def forward_train(self, x: torch.Tensor, mask: torch.Tensor):
        hw = x.shape[-2:]
        _, _, c3, c4 = self.backbone(x)
        logits = self._decode(c3, c4, hw)
        aux_logits = det_interpolate(self.aux_head(c3), size=hw)
        main_loss, terms = standard_sem_loss(logits, mask)
        aux_ce = F.cross_entropy(aux_logits, mask, reduction='none').mean()
        total = main_loss + self.aux_weight * aux_ce
        terms['loss_aux'] = aux_ce
        terms['loss_total'] = total
        return logits, terms


# -----------------------------------------------------------------------------
# 4. DeepLabv3+ (Chen et al., ECCV 2018) - ResNet-50 OS=16, ASPP (6,12,18), 48-ch low-level, ~39.76M
# -----------------------------------------------------------------------------

class ASPP(nn.Module):
    def __init__(self, in_ch: int = 2048, out_ch: int = 256, rates=(6, 12, 18)):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=rates[0], dilation=rates[0], bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=rates[1], dilation=rates[1], bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=rates[2], dilation=rates[2], bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.pool = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        p = self.pool(x.mean(dim=(-2, -1), keepdim=True)).expand(-1, -1, h, w)
        return self.proj(torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), p], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = ResNet50Backbone(replace_stride_with_dilation=(False, False, True), deep_stem=False)
        self.aspp = ASPP(2048, 256, rates=(6, 12, 18))
        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        c1, _, _, c4 = self.backbone(x)
        low = self.low_proj(c1)
        high = det_interpolate(self.aspp(c4), size=low.shape[-2:])
        out = self.decoder(torch.cat([high, low], dim=1))
        return det_interpolate(out, size=hw)

    def forward_train(self, x: torch.Tensor, mask: torch.Tensor):
        logits = self.forward(x)
        _, terms = standard_sem_loss(logits, mask)
        return logits, terms


# -----------------------------------------------------------------------------
# 5. HRNetV2-W48 (Wang et al., TPAMI 2021) - seg_hrnet_w48, ~65.85M
# -----------------------------------------------------------------------------

class HRBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class HRModule(nn.Module):
    def __init__(self, num_branches: int, num_blocks, num_channels):
        super().__init__()
        self.num_branches = num_branches
        self.num_channels = num_channels
        self.branches = nn.ModuleList([
            nn.Sequential(*[HRBasicBlock(num_channels[i], num_channels[i]) for _ in range(num_blocks[i])])
            for i in range(num_branches)
        ])
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(num_channels[i]),
                    ))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    convs = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            convs.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[i], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_channels[i]),
                            ))
                        else:
                            convs.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_channels[j]),
                                nn.ReLU(inplace=True),
                            ))
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        self.fuse_layers = nn.ModuleList(fuse_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            target_hw = x[i].shape[-2:]
            for j in range(self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + det_interpolate(self.fuse_layers[i][j](x[j]), size=target_hw)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNetW48(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        ds1 = nn.Sequential(nn.Conv2d(64, 256, 1, bias=False), nn.BatchNorm2d(256))
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=ds1),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        c2 = [48, 96]
        self.transition1 = self._make_transition([256], c2)
        self.stage2 = nn.ModuleList([HRModule(2, [4, 4], c2) for _ in range(1)])

        c3 = [48, 96, 192]
        self.transition2 = self._make_transition(c2, c3)
        self.stage3 = nn.ModuleList([HRModule(3, [4, 4, 4], c3) for _ in range(4)])

        c4 = [48, 96, 192, 384]
        self.transition3 = self._make_transition(c3, c4)
        self.stage4 = nn.ModuleList([HRModule(4, [4, 4, 4, 4], c4) for _ in range(3)])

        total_ch = sum(c4)  # 720
        self.last_layer = nn.Sequential(
            nn.Conv2d(total_ch, total_ch, 1, bias=False),
            nn.BatchNorm2d(total_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_ch, num_classes, 1),
        )

    @staticmethod
    def _make_transition(pre_ch, cur_ch):
        num_pre = len(pre_ch)
        num_cur = len(cur_ch)
        layers = []
        for i in range(num_cur):
            if i < num_pre:
                if cur_ch[i] != pre_ch[i]:
                    layers.append(nn.Sequential(
                        nn.Conv2d(pre_ch[i], cur_ch[i], 3, padding=1, bias=False),
                        nn.BatchNorm2d(cur_ch[i]),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    layers.append(nn.Identity())
            else:
                convs = []
                for j in range(i + 1 - num_pre):
                    inch = pre_ch[-1]
                    outch = cur_ch[i] if j == i - num_pre else inch
                    convs.append(nn.Sequential(
                        nn.Conv2d(inch, outch, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(outch),
                        nn.ReLU(inplace=True),
                    ))
                layers.append(nn.Sequential(*convs))
        return nn.ModuleList(layers)

    @staticmethod
    def _apply_transition(trans, x_list):
        out = []
        for i, layer in enumerate(trans):
            out.append(layer(x_list[i] if i < len(x_list) else x_list[-1]))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        xs = self._apply_transition(self.transition1, [x])
        for mod in self.stage2:
            xs = mod(xs)
        xs = self._apply_transition(self.transition2, xs)
        for mod in self.stage3:
            xs = mod(xs)
        xs = self._apply_transition(self.transition3, xs)
        for mod in self.stage4:
            xs = mod(xs)

        h0, w0 = xs[0].shape[-2:]
        feats = [xs[0]] + [det_interpolate(f, size=(h0, w0)) for f in xs[1:]]
        logits = self.last_layer(torch.cat(feats, dim=1))
        return det_interpolate(logits, size=hw)

    def forward_train(self, x: torch.Tensor, mask: torch.Tensor):
        logits = self.forward(x)
        _, terms = standard_sem_loss(logits, mask)
        return logits, terms


# -----------------------------------------------------------------------------
# 6. SegFormer-B1 (Xie et al., NeurIPS 2021) - MiT-B1 + All-MLP (256), ~13.68M
# -----------------------------------------------------------------------------

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        b, c, h, w = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, h, w


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = det_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = self.fc1(x)
        ch = x.shape[-1]
        x = self.dwconv(x.transpose(1, 2).reshape(b, ch, h, w)).flatten(2).transpose(1, 2)
        return self.fc2(self.act(x))


class MiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, sr_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class SegFormerB1(nn.Module):
    def __init__(self, num_classes: int = 4, embed_dims=(64, 128, 320, 512), depths=(2, 2, 2, 2),
                 num_heads=(1, 2, 5, 8), sr_ratios=(8, 4, 2, 1), decoder_dim: int = 256):
        super().__init__()
        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(3 if i == 0 else embed_dims[i - 1], embed_dims[i],
                              patch_size=7 if i == 0 else 3, stride=4 if i == 0 else 2)
            for i in range(4)
        ])
        self.blocks = nn.ModuleList([
            nn.ModuleList([MiTBlock(embed_dims[i], num_heads[i], 4, sr_ratios[i]) for _ in range(depths[i])])
            for i in range(4)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims[i]) for i in range(4)])

        self.linear_c = nn.ModuleList([nn.Linear(embed_dims[i], decoder_dim) for i in range(4)])
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        b = x.shape[0]
        feats = []
        cur = x
        for pe, blks, norm in zip(self.patch_embeds, self.blocks, self.norms):
            cur, h, w = pe(cur)
            for blk in blks:
                cur = blk(cur, h, w)
            cur = norm(cur).transpose(1, 2).reshape(b, -1, h, w).contiguous()
            feats.append(cur)

        h0, w0 = feats[0].shape[-2:]
        proj_feats = []
        for f, lin in zip(feats, self.linear_c):
            b, c, h, w = f.shape
            p = lin(f.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, -1, h, w)
            proj_feats.append(det_interpolate(p, size=(h0, w0)))
        logits = self.linear_fuse(torch.cat(proj_feats, dim=1))
        return det_interpolate(logits, size=hw)

    def forward_train(self, x: torch.Tensor, mask: torch.Tensor):
        logits = self.forward(x)
        _, terms = standard_sem_loss(logits, mask)
        return logits, terms


# -----------------------------------------------------------------------------
# 7. Mask2Former (Cheng et al., CVPR 2022) - ResNet-50 + 6-layer MSDeformAttn + 9-layer Masked Decoder (~44.0M)
# -----------------------------------------------------------------------------

def sine_pos_embed_2d(h: int, w: int, num_feats: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """2D Sine positional encoding returning (1, 2*num_feats, h, w)."""
    y_embed = torch.arange(1, h + 1, dtype=dtype, device=device).unsqueeze(1).expand(h, w) / (h + 1e-6) * (2.0 * math.pi)
    x_embed = torch.arange(1, w + 1, dtype=dtype, device=device).unsqueeze(0).expand(h, w) / (w + 1e-6) * (2.0 * math.pi)
    dim_t = 10000.0 ** (2 * (torch.arange(num_feats, dtype=dtype, device=device) // 2) / num_feats)
    pos_x = x_embed.unsqueeze(-1) / dim_t
    pos_y = y_embed.unsqueeze(-1) / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1).permute(2, 0, 1).unsqueeze(0)
    return pos


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention in pure PyTorch using deterministic det_grid_sample."""
    def __init__(self, d_model: int = 256, n_levels: int = 3, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor, input_flatten: torch.Tensor, spatial_shapes):
        """
        query: (B, Nq, C)
        reference_points: (B, Nq, L, 2) in [0, 1]
        input_flatten: (B, Nv, C)
        spatial_shapes: list of (H_l, W_l)
        """
        B, Nq, C = query.shape
        Nv = input_flatten.shape[1]
        head_dim = C // self.n_heads
        value = self.value_proj(input_flatten).view(B, Nv, self.n_heads, head_dim)
        sampling_offsets = self.sampling_offsets(query).view(B, Nq, self.n_heads, self.n_levels, self.n_points, 2)
        attn_weights = self.attention_weights(query).view(B, Nq, self.n_heads, self.n_levels * self.n_points).softmax(-1)
        attn_weights = attn_weights.view(B, Nq, self.n_heads, self.n_levels, self.n_points)

        offset_normalizer = torch.tensor([[w, h] for h, w in spatial_shapes], dtype=query.dtype, device=query.device)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        sampling_grids = 2.0 * sampling_locations - 1.0  # (B, Nq, H_head, L, P, 2)

        value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
        sampling_value_list = []
        for lid, (H_l, W_l) in enumerate(spatial_shapes):
            val_l = value_list[lid].permute(0, 2, 3, 1).reshape(B * self.n_heads, head_dim, H_l, W_l)
            grid_l = sampling_grids[:, :, :, lid].permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, Nq, self.n_points, 2)
            sampled_l = det_grid_sample(val_l, grid_l)  # (B*n_heads, head_dim, Nq, n_points)
            sampling_value_list.append(sampled_l)

        sampling_values = torch.stack(sampling_value_list, dim=-2)  # (B*n_heads, head_dim, Nq, L, P)
        attn = attn_weights.permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, 1, Nq, self.n_levels, self.n_points)
        output = (sampling_values * attn).sum(dim=(-2, -1)).view(B, self.n_heads * head_dim, Nq).transpose(1, 2).contiguous()
        return self.output_proj(output)


class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, d_ffn: int = 1024, n_levels: int = 3, n_heads: int = 8, n_points: int = 4, use_time_mlp: bool = False, time_dim: int = 1024):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, d_model * 2)) if use_time_mlp else None

    def forward(self, src: torch.Tensor, pos: torch.Tensor, reference_points: torch.Tensor, spatial_shapes, time_emb: torch.Tensor = None):
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes)
        src = self.norm1(src + src2)
        if self.time_mlp is not None and time_emb is not None:
            scale, shift = self.time_mlp(time_emb).unsqueeze(1).chunk(2, dim=-1)
            src = src * (scale + 1.0) + shift
        src2 = self.linear2(self.act(self.linear1(src)))
        return self.norm2(src + src2)


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(self, in_channels=(256, 512, 1024, 2048), conv_dim: int = 256, num_layers: int = 6):
        super().__init__()
        # 3 transformer levels: 1/8 (512), 1/16 (1024), 1/32 (2048)
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, conv_dim, 1), nn.GroupNorm(32, conv_dim))
            for ch in in_channels[1:]
        ])
        self.level_embed = nn.Parameter(torch.zeros(3, conv_dim))
        nn.init.normal_(self.level_embed)
        self.encoder_layers = nn.ModuleList([
            DeformableEncoderLayer(conv_dim, d_ffn=1024, n_levels=3, n_heads=8, n_points=4)
            for _ in range(num_layers)
        ])
        # Lateral connection for 1/4 feature (256 ch)
        self.lateral_conv = nn.Sequential(nn.Conv2d(in_channels[0], conv_dim, 1, bias=False), nn.GroupNorm(32, conv_dim))
        self.output_conv = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, conv_dim),
            nn.ReLU(inplace=True),
        )
        self.mask_features = nn.Conv2d(conv_dim, conv_dim, 1)

    @staticmethod
    def _reference_points(spatial_shapes, device: torch.device, dtype: torch.dtype):
        refs = []
        for H_l, W_l in spatial_shapes:
            ry, rx = torch.meshgrid(
                torch.linspace(0.5, H_l - 0.5, H_l, dtype=dtype, device=device) / H_l,
                torch.linspace(0.5, W_l - 0.5, W_l, dtype=dtype, device=device) / W_l,
                indexing='ij',
            )
            refs.append(torch.stack((rx.reshape(-1), ry.reshape(-1)), dim=-1))
        ref = torch.cat(refs, dim=0)  # (Nv, 2)
        return ref[None, :, None, :].expand(1, -1, len(spatial_shapes), 2)

    def forward(self, features):
        c1, c2, c3, c4 = features  # 1/4, 1/8, 1/16, 1/32
        srcs, poses, spatial_shapes = [], [], []
        for idx, (f, proj) in enumerate(zip((c2, c3, c4), self.input_proj)):
            src = proj(f)
            b, c, h, w = src.shape
            spatial_shapes.append((h, w))
            pos = sine_pos_embed_2d(h, w, c // 2, src.device, src.dtype).expand(b, -1, -1, -1)
            srcs.append(src.flatten(2).transpose(1, 2))
            poses.append(pos.flatten(2).transpose(1, 2) + self.level_embed[idx].view(1, 1, -1))

        src_flatten = torch.cat(srcs, dim=1)
        pos_flatten = torch.cat(poses, dim=1)
        ref_pts = self._reference_points(spatial_shapes, src_flatten.device, src_flatten.dtype).expand(src_flatten.shape[0], -1, -1, -1)

        memory = src_flatten
        for layer in self.encoder_layers:
            memory = run_ckpt(layer, memory, pos_flatten, ref_pts, spatial_shapes)

        b = c1.shape[0]
        outs = memory.split([h * w for h, w in spatial_shapes], dim=1)
        multi_scale = [
            o.transpose(1, 2).reshape(b, -1, h, w).contiguous()
            for o, (h, w) in zip(outs, spatial_shapes)
        ]  # [1/8, 1/16, 1/32]
        y = self.lateral_conv(c1) + det_interpolate(multi_scale[0], size=c1.shape[-2:])
        y = self.output_conv(y)
        mask_feat = self.mask_features(y)  # (B, 256, H/4, W/4)
        # Return mask_features and [1/32, 1/16, 1/8] for the Masked Transformer Decoder
        return mask_feat, [multi_scale[2], multi_scale[1], multi_scale[0]]


class MaskedTransformerDecoderLayer(nn.Module):
    """Mask2Former decoder layer: Masked Cross-Attention -> Self-Attention -> FFN."""
    def __init__(self, d_model: int = 256, n_heads: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.ca_q = nn.Linear(d_model, d_model)
        self.ca_k = nn.Linear(d_model, d_model)
        self.ca_v = nn.Linear(d_model, d_model)
        self.ca_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        self.sa_qkv = nn.Linear(d_model, d_model * 3)
        self.sa_proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.act = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, attn_mask: torch.Tensor, pos: torch.Tensor, query_pos: torch.Tensor):
        B, Q, C = tgt.shape
        Nk = memory.shape[1]
        hd = C // self.n_heads
        # 1. Masked Cross-Attention
        q = self.ca_q(tgt + query_pos).view(B, Q, self.n_heads, hd).permute(0, 2, 1, 3)
        k = self.ca_k(memory + pos).view(B, Nk, self.n_heads, hd).permute(0, 2, 1, 3)
        v = self.ca_v(memory).view(B, Nk, self.n_heads, hd).permute(0, 2, 1, 3)
        ca_out = det_attention(q, k, v, attn_mask=attn_mask).permute(0, 2, 1, 3).reshape(B, Q, C)
        tgt = self.norm1(tgt + self.ca_proj(ca_out))

        # 2. Self-Attention
        qkv = self.sa_qkv(tgt + query_pos).view(B, Q, 3, self.n_heads, hd).permute(2, 0, 3, 1, 4)
        sa_out = det_attention(qkv[0], qkv[1], qkv[2]).permute(0, 2, 1, 3).reshape(B, Q, C)
        tgt = self.norm2(tgt + self.sa_proj(sa_out))

        # 3. FFN
        tgt2 = self.linear2(self.act(self.linear1(tgt)))
        return self.norm3(tgt + tgt2)


def _exact_bipartite_match(cost_matrix: np.ndarray):
    """Exact minimum-cost bipartite matching for Q x M where M <= 4 (pure Python/NumPy, no scipy needed)."""
    Q, M = cost_matrix.shape
    if M == 0:
        return [], []
    # Only the top-M smallest rows per column can ever be part of the optimal matching
    cand_rows = set()
    for c in range(M):
        order = np.argsort(cost_matrix[:, c], kind='mergesort')[:M]
        for r in order:
            cand_rows.add(int(r))
    cand_rows = sorted(cand_rows)
    best_cost = float('inf')
    best_rows = None
    for rows in itertools.permutations(cand_rows, M):
        s = sum(float(cost_matrix[rows[c], c]) for c in range(M))
        if s < best_cost:
            best_cost = s
            best_rows = rows
    return list(best_rows), list(range(M))


class Mask2Former(nn.Module):
    def __init__(self, num_classes: int = 4, num_queries: int = 100, dec_layers: int = 9,
                 cost_class: float = 2.0, cost_mask: float = 5.0, cost_dice: float = 5.0,
                 no_object_weight: float = 0.1, num_points: int = 12544):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.dec_layers = dec_layers
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.no_object_weight = no_object_weight
        self.num_points = num_points

        self.backbone = ResNet50Backbone(replace_stride_with_dilation=(False, False, False), deep_stem=False)
        self.pixel_decoder = MSDeformAttnPixelDecoder((256, 512, 1024, 2048), conv_dim=256, num_layers=6)

        d_model = 256
        self.query_feat = nn.Embedding(num_queries, d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.level_embed = nn.Embedding(3, d_model)
        self.decoder_layers = nn.ModuleList([
            MaskedTransformerDecoderLayer(d_model=d_model, n_heads=8, dim_feedforward=2048)
            for _ in range(dec_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = no_object_weight
        self.register_buffer('empty_weight', empty_weight)

    def _predict_head(self, output: torch.Tensor, mask_features: torch.Tensor, attn_target_size):
        out_norm = self.decoder_norm(output)
        outputs_class = self.class_embed(out_norm)
        mask_embed = self.mask_embed(out_norm)
        outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
        attn_mask = det_interpolate(outputs_mask, size=attn_target_size)
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).expand(-1, 8, -1, -1) < 0.5).detach()
        # Prevent all-True masked rows (attend everywhere if mask is empty)
        all_masked = attn_mask.all(dim=-1, keepdim=True)
        attn_mask = attn_mask & (~all_masked)
        return outputs_class, outputs_mask, attn_mask

    def _forward_heads(self, x: torch.Tensor):
        c1, c2, c3, c4 = self.backbone(x)
        mask_features, ms_feats = self.pixel_decoder((c1, c2, c3, c4))  # ms_feats: [1/32, 1/16, 1/8]
        B = x.shape[0]
        src_list, pos_list, size_list = [], [], []
        for i in range(3):
            _, c, h, w = ms_feats[i].shape
            size_list.append((h, w))
            pos = sine_pos_embed_2d(h, w, c // 2, x.device, x.dtype).expand(B, -1, -1, -1).flatten(2).transpose(1, 2)
            src = ms_feats[i].flatten(2).transpose(1, 2) + self.level_embed.weight[i].view(1, 1, -1)
            src_list.append(src)
            pos_list.append(pos)

        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        output = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)

        predictions_class, predictions_mask = [], []
        cls0, msk0, attn_mask = self._predict_head(output, mask_features, size_list[0])
        predictions_class.append(cls0)
        predictions_mask.append(msk0)

        for i, layer in enumerate(self.decoder_layers):
            lvl = i % 3
            output = layer(output, src_list[lvl], attn_mask, pos_list[lvl], query_embed)
            cls_i, msk_i, attn_mask = self._predict_head(output, mask_features, size_list[(i + 1) % 3])
            predictions_class.append(cls_i)
            predictions_mask.append(msk_i)

        return predictions_class, predictions_mask

    @staticmethod
    def _semantic_logits(pred_cls: torch.Tensor, pred_mask: torch.Tensor, hw) -> torch.Tensor:
        cls_prob = pred_cls.softmax(-1)[..., :-1]  # (B, Q, 4)
        mask_prob = pred_mask.sigmoid()            # (B, Q, H/4, W/4)
        sem_prob = torch.einsum('bqc,bqhw->bchw', cls_prob, mask_prob)
        sem_prob = det_interpolate(sem_prob, size=hw)
        sem_prob = sem_prob.clamp_min(1e-6)
        sem_prob = sem_prob / sem_prob.sum(dim=1, keepdim=True)
        return sem_prob.log()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds_cls, preds_mask = self._forward_heads(x)
        return self._semantic_logits(preds_cls[-1], preds_mask[-1], x.shape[-2:])

    def _sample_points_uncertainty(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor):
        """Importance point sampling (12544 points, oversample=3.0, importance_ratio=0.75) via det_grid_sample."""
        M, H, W = pred_masks.shape
        if M == 0:
            return pred_masks.new_zeros((0, self.num_points)), gt_masks.new_zeros((0, self.num_points))
        num_pts = self.num_points
        num_oversampled = int(num_pts * 3.0)
        num_uncertain = int(0.75 * num_pts)
        num_random = num_pts - num_uncertain

        coords = torch.rand(M, num_oversampled, 1, 2, device=pred_masks.device, dtype=pred_masks.dtype) * 2.0 - 1.0
        sampled_logits = det_grid_sample(pred_masks.unsqueeze(1), coords).squeeze(1).squeeze(-1)  # (M, num_oversampled)
        uncertainty = -sampled_logits.abs()
        _, idx = torch.topk(uncertainty, k=num_uncertain, dim=1)
        shift = torch.arange(M, device=pred_masks.device).unsqueeze(1) * num_oversampled
        uncertain_coords = coords.view(-1, 1, 2)[(idx + shift).view(-1)].view(M, num_uncertain, 1, 2)
        rand_coords = torch.rand(M, num_random, 1, 2, device=pred_masks.device, dtype=pred_masks.dtype) * 2.0 - 1.0
        point_coords = torch.cat([uncertain_coords, rand_coords], dim=1)  # (M, num_pts, 1, 2)

        pred_pts = det_grid_sample(pred_masks.unsqueeze(1), point_coords).squeeze(1).squeeze(-1)
        gt_pts = det_grid_sample(gt_masks.unsqueeze(1), point_coords).squeeze(1).squeeze(-1)
        return pred_pts, gt_pts

    def forward_train(self, x: torch.Tensor, mask: torch.Tensor):
        hw = x.shape[-2:]
        preds_cls, preds_mask = self._forward_heads(x)
        B, Q = preds_cls[-1].shape[:2]

        # Per-image GT binary masks at full input resolution; the original matcher and criterion
        # point-sample full-resolution targets with the same normalized coordinates as the 1/4 mask logits.
        onehot = F.one_hot(mask, self.num_classes).permute(0, 3, 1, 2).float()
        targets = []
        total_num_masks = 0
        for b in range(B):
            present = [c for c in range(self.num_classes) if bool((mask[b] == c).any())]
            gt_labels = torch.tensor(present, dtype=torch.long, device=x.device)
            gt_m = onehot[b, present]
            targets.append((gt_labels, gt_m))
            total_num_masks += len(present)
        total_num_masks = max(total_num_masks, 1)

        loss_cls_sum = x.new_zeros(())
        loss_mask_sum = x.new_zeros(())
        loss_dice_sum = x.new_zeros(())
        n_layers = len(preds_cls)

        for l_cls, l_mask in zip(preds_cls, preds_mask):
            # 1. Bipartite matching per image
            indices = []
            with torch.no_grad():
                out_prob = l_cls.softmax(-1)
                for b in range(B):
                    gt_labels, gt_m = targets[b]
                    M = len(gt_labels)
                    if M == 0:
                        indices.append(([], []))
                        continue
                    cost_c = -out_prob[b][:, gt_labels]  # (Q, M)
                    pts = torch.rand(1, self.num_points, 1, 2, device=x.device, dtype=x.dtype) * 2.0 - 1.0
                    p_pts = det_grid_sample(l_mask[b].unsqueeze(0), pts).squeeze(0).squeeze(-1)  # (Q, P)
                    g_pts = det_grid_sample(gt_m.unsqueeze(0), pts).squeeze(0).squeeze(-1)       # (M, P)
                    # Pairwise sigmoid CE cost
                    pos = F.softplus(-p_pts)
                    neg = p_pts + pos
                    cost_m = (torch.matmul(pos, g_pts.t()) + torch.matmul(neg, (1.0 - g_pts).t())) / self.num_points
                    # Pairwise dice cost
                    p_sig = p_pts.sigmoid()
                    num = 2.0 * torch.matmul(p_sig, g_pts.t())
                    den = p_sig.sum(-1, keepdim=True) + g_pts.sum(-1).unsqueeze(0)
                    cost_d = 1.0 - (num + 1.0) / (den + 1.0)
                    cost = (self.cost_class * cost_c + self.cost_mask * cost_m + self.cost_dice * cost_d).cpu().numpy()
                    indices.append(_exact_bipartite_match(cost))

            # 2. Classification loss
            target_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=x.device)
            pred_matched, gt_matched = [], []
            for b, (src_idx, tgt_idx) in enumerate(indices):
                if len(src_idx):
                    target_classes[b, src_idx] = targets[b][0][tgt_idx]
                    pred_matched.append(l_mask[b, src_idx])
                    gt_matched.append(targets[b][1][tgt_idx])

            # Weighted-mean CE (= F.cross_entropy(..., weight=empty_weight) in the original SetCriterion),
            # written as weighted sum / weight sum because mean-reduced CE has no deterministic CUDA kernel.
            ce_q = F.cross_entropy(l_cls.transpose(1, 2), target_classes, weight=self.empty_weight, reduction='none')
            loss_cls_sum = loss_cls_sum + self.cost_class * (ce_q.sum() / self.empty_weight[target_classes].sum())

            if len(pred_matched):
                pm = torch.cat(pred_matched, dim=0)
                gm = torch.cat(gt_matched, dim=0)
                p_pts, g_pts = self._sample_points_uncertainty(pm, gm)
                bce = F.binary_cross_entropy_with_logits(p_pts, g_pts, reduction='none').mean(1).sum() / total_num_masks
                p_sig = p_pts.sigmoid()
                inter = 2.0 * (p_sig * g_pts).sum(1)
                denom = p_sig.sum(1) + g_pts.sum(1)
                d_loss = (1.0 - (inter + 1.0) / (denom + 1.0)).sum() / total_num_masks
                loss_mask_sum = loss_mask_sum + self.cost_mask * bce
                loss_dice_sum = loss_dice_sum + self.cost_dice * d_loss

        # Original Mask2Former objective: 2*CE + 5*BCE + 5*Dice summed over the initial prediction and all
        # 9 decoder layers (deep supervision); no extra per-pixel semantic loss (revision 2026-10-23).
        total = loss_cls_sum + loss_mask_sum + loss_dice_sum
        with torch.no_grad():
            logits = self._semantic_logits(preds_cls[-1], preds_mask[-1], hw)
        terms = {
            'loss_total': total,
            'loss_m2f_cls': loss_cls_sum / n_layers,
            'loss_m2f_mask': loss_mask_sum / n_layers,
            'loss_m2f_dice': loss_dice_sum / n_layers,
        }
        return logits, terms


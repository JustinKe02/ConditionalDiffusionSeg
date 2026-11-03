"""Diffusion segmentation baselines: DDP (Ji et al., ICCV 2023) and CCDM (Zbinden et al., ICCV 2023).

Implemented in pure PyTorch with full CUDA determinism.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import (
    ResNet50Backbone,
    DeformableEncoderLayer,
    det_interpolate,
    det_attention,
    sine_pos_embed_2d,
    run_ckpt,
    standard_sem_loss,
)


# =============================================================================
# 8. DDP: Diffusion Model for Dense Visual Prediction (Ji et al., ICCV 2023)
# =============================================================================

def alpha_cosine_log_snr(t: torch.Tensor, ns: float = 0.0002, ds: float = 0.00025) -> torch.Tensor:
    val = (torch.cos((t + ns) / (1.0 + ds) * math.pi * 0.5) ** -2) - 1.0
    return -torch.log(val.clamp(min=1e-5))


def log_snr_to_alpha_sigma(log_snr: torch.Tensor):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


class LearnedSinusoidalPosEmb(nn.Module):
    """Official DDP learned sinusoidal positional embedding (ddp.py lines 778-793)."""
    def __init__(self, dim: int = 16):
        super().__init__()
        assert (dim % 2) == 0
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        freqs = x * self.weights.view(1, -1) * (2.0 * math.pi)
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


class FPNMergeNeck(nn.Module):
    """Official DDP neck: FPN (4 levels -> 256, GN32) + MultiStageMerging (concat at 1/4 -> 256, GN32)."""
    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_channels, 1, bias=False), nn.GroupNorm(32, out_channels))
            for c in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.GroupNorm(32, out_channels))
            for _ in in_channels
        ])
        self.merge = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels), out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
        )

    def forward(self, feats):
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + det_interpolate(laterals[i], size=laterals[i - 1].shape[-2:])
        outs = [fpn(lat) for fpn, lat in zip(self.fpn_convs, laterals)]
        h0, w0 = outs[0].shape[-2:]
        aligned = [outs[0]] + [det_interpolate(o, size=(h0, w0)) for o in outs[1:]]
        return self.merge(torch.cat(aligned, dim=1))


class DeformableHeadWithTime(nn.Module):
    """Official DDP decode head: 6-layer Deformable Transformer Encoder with time modulation (num_levels=1)."""
    def __init__(self, in_channels: int = 256, num_classes: int = 4, num_layers: int = 6, time_dim: int = 1024):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableEncoderLayer(
                d_model=in_channels,
                d_ffn=1024,
                n_levels=1,
                n_heads=8,
                n_points=4,
                use_time_mlp=True,
                time_dim=time_dim,
            )
            for _ in range(num_layers)
        ])
        self.conv_seg = nn.Conv2d(in_channels, num_classes, 1)

    @staticmethod
    def _ref_points_1level(h: int, w: int, device: torch.device, dtype: torch.dtype):
        ry, rx = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, dtype=dtype, device=device) / h,
            torch.linspace(0.5, w - 0.5, w, dtype=dtype, device=device) / w,
            indexing='ij',
        )
        ref = torch.stack((rx.reshape(-1), ry.reshape(-1)), dim=-1)
        return ref[None, :, None, :]  # (1, H*W, 1, 2)

    def forward(self, feat: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        pos = sine_pos_embed_2d(h, w, c // 2, feat.device, feat.dtype).expand(b, -1, -1, -1).flatten(2).transpose(1, 2)
        src = feat.flatten(2).transpose(1, 2)
        ref_pts = self._ref_points_1level(h, w, feat.device, feat.dtype).expand(b, -1, -1, -1)
        spatial_shapes = [(h, w)]
        memory = src
        for layer in self.layers:
            memory = run_ckpt(layer, memory, pos, ref_pts, spatial_shapes, time_emb)
        memory = memory.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return self.conv_seg(memory)


class DDP(nn.Module):
    """DDP segmentor (Ji et al., ICCV 2023) with ResNet-50 + FPN + DeformableHeadWithTime."""
    is_diffusion = True
    noise_channels = 256
    noise_down_factor = 4

    def __init__(self, num_classes: int = 4, bit_scale: float = 0.01, timesteps: int = 3,
                 sample_range=(0.0, 0.999), time_difference: float = 1.0, aux_weight: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.bit_scale = bit_scale
        self.timesteps = timesteps
        self.sample_range = sample_range
        self.time_difference = time_difference
        self.aux_weight = aux_weight

        feat_dim = 256
        self.backbone = ResNet50Backbone(replace_stride_with_dilation=(False, False, False), deep_stem=False)
        self.neck = FPNMergeNeck((256, 512, 1024, 2048), out_channels=feat_dim)
        self.embedding_table = nn.Embedding(num_classes + 1, feat_dim)
        self.transform = nn.Conv2d(feat_dim * 2, feat_dim, 1)

        time_dim = feat_dim * 4  # 1024
        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(16),
            nn.Linear(17, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.decode_head = DeformableHeadWithTime(feat_dim, num_classes=num_classes, num_layers=6, time_dim=time_dim)
        self.aux_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(feat_dim, num_classes, 1),
        )

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        return self.neck(self.backbone(x))

    def _encode_mask(self, mask_idx: torch.Tensor) -> torch.Tensor:
        emb = self.embedding_table(mask_idx).permute(0, 3, 1, 2)
        return (torch.sigmoid(emb) * 2.0 - 1.0) * self.bit_scale

    def forward_train(self, x: torch.Tensor, mask: torch.Tensor):
        hw = x.shape[-2:]
        feat_x = self.extract_feat(x)  # (B, 256, H/4, W/4)
        b, c, h, w = feat_x.shape
        mask_down = F.interpolate(mask.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(1).long()
        gt_down = self._encode_mask(mask_down)

        times = torch.empty((b,), device=x.device, dtype=x.dtype).uniform_(self.sample_range[0], self.sample_range[1])
        noise = torch.randn_like(gt_down)
        log_snr = alpha_cosine_log_snr(times)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr[:, None, None, None])
        noised_gt = alpha * gt_down + sigma * noise

        feat = self.transform(torch.cat([feat_x, noised_gt], dim=1))
        input_times = self.time_mlp(log_snr)
        logits_1_4 = self.decode_head(feat, input_times)
        logits = det_interpolate(logits_1_4, size=hw)

        aux_logits = det_interpolate(self.aux_head(feat_x), size=hw)
        main_loss, terms = standard_sem_loss(logits, mask)
        aux_ce = F.cross_entropy(aux_logits, mask, reduction='none').mean()
        total = main_loss + self.aux_weight * aux_ce
        terms['loss_aux'] = aux_ce
        terms['loss_total'] = total
        return logits, terms

    def _sampling_timesteps(self, batch: int, steps: int, device: torch.device, dtype: torch.dtype):
        pairs = []
        for step in range(steps):
            t_now = 1.0 - (step / steps) * (1.0 - self.sample_range[0])
            t_next = max(1.0 - (step + 1.0 + self.time_difference) / steps * (1.0 - self.sample_range[0]), self.sample_range[0])
            pairs.append((
                torch.full((batch,), t_now, device=device, dtype=dtype),
                torch.full((batch,), t_next, device=device, dtype=dtype),
            ))
        return pairs

    @torch.no_grad()
    def infer_windows(self, x: torch.Tensor, eps: torch.Tensor = None, steps: int = None):
        steps = self.timesteps if steps is None else steps
        hw = x.shape[-2:]
        feat_x = self.extract_feat(x)
        b, c, h, w = feat_x.shape
        mask_t = torch.randn((b, c, h, w), device=x.device, dtype=x.dtype) if eps is None else eps
        time_pairs = self._sampling_timesteps(b, steps, x.device, x.dtype)

        mask_logit = None
        for times_now, times_next in time_pairs:
            feat = self.transform(torch.cat([feat_x, mask_t], dim=1))
            log_snr = alpha_cosine_log_snr(times_now)
            log_snr_next = alpha_cosine_log_snr(times_next)
            alpha, sigma = log_snr_to_alpha_sigma(log_snr[:, None, None, None])
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next[:, None, None, None])

            input_times = self.time_mlp(log_snr)
            mask_logit = self.decode_head(feat, input_times)
            mask_pred = self._encode_mask(mask_logit.argmax(dim=1))
            pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8)
            mask_t = mask_pred * alpha_next + pred_noise * sigma_next

        return det_interpolate(mask_logit, size=hw), b * steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.infer_windows(x)
        return logits


# =============================================================================
# 9. CCDM: Conditional Categorical Diffusion Models (Zbinden et al., ICCV 2023)
# =============================================================================

def ccdm_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def sample_categorical_det(probs: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
    """Deterministic 4-class one-hot categorical sampling avoiding CUDA cumsum/multinomial."""
    p = probs.clamp_min(1e-12)
    p = p / p.sum(dim=1, keepdim=True)
    if u is None:
        u = torch.rand(p.shape[0], p.shape[2], p.shape[3], device=p.device, dtype=p.dtype)
    c0 = p[:, 0]
    c1 = c0 + p[:, 1]
    c2 = c1 + p[:, 2]
    cls = (u > c0).long() + (u > c1).long() + (u > c2).long()
    return F.one_hot(cls, 4).permute(0, 3, 1, 2).float()


class CCDMResBlock(nn.Module):
    """OpenAI ADM U-Net residual block conditioned on timestep embedding (unet.py lines 139-250)."""
    def __init__(self, in_ch: int, emb_ch: int, out_ch: int = None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.in_ch, self.out_ch = in_ch, out_ch
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32 if in_ch >= 32 else 8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_ch, out_ch),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32 if out_ch >= 32 else 8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        nn.init.zeros_(self.out_layers[-1].weight)
        nn.init.zeros_(self.out_layers[-1].bias)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = h + self.emb_layers(emb)[:, :, None, None]
        h = self.out_layers(h)
        return self.skip(x) + h


class CCDMAttentionBlock(nn.Module):
    """OpenAI ADM spatial self-attention block at inner resolutions (unet.py lines 253-305)."""
    def __init__(self, channels: int, num_head_channels: int = 32):
        super().__init__()
        self.num_heads = max(1, channels // num_head_channels)
        self.norm = nn.GroupNorm(32 if channels >= 32 else 8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        hd = c // self.num_heads
        qkv = self.qkv(self.norm(x).reshape(b, c, n))
        q, k, v = qkv.reshape(b, self.num_heads, 3 * hd, n).chunk(3, dim=2)
        out = det_attention(q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1))
        out = out.transpose(-2, -1).reshape(b, c, n)
        return (x.reshape(b, c, n) + self.proj_out(out)).reshape(b, c, h, w)


class CCDMUNet(nn.Module):
    """OpenAI ADM U-Net for CCDM (ddpm/models/unet_openai):
    base_channels=64, channel_mult=(0.5, 1, 1, 2, 2, 4, 4) for 512x512 -> [32, 64, 64, 128, 128, 256, 256],
    num_res_blocks=2, self-attention at the 3 innermost scales (ds in [16, 32, 64], i.e. 32x32, 16x16, 8x8).
    """
    def __init__(self, in_channels: int = 7, base_channels: int = 64, out_channels: int = 4,
                 channel_mult=(0.5, 1, 1, 2, 2, 4, 4), num_res_blocks: int = 2,
                 attn_ds=(16, 32, 64), num_head_channels: int = 32):
        super().__init__()
        self.base_channels = base_channels
        emb_ch = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )

        ch0 = int(channel_mult[0] * base_channels)
        self.input_conv = nn.Conv2d(in_channels, ch0, 3, padding=1)
        input_block_chans = [ch0]
        ch = ch0
        ds = 1

        self.down_blocks = nn.ModuleList()
        for level, mult in enumerate(channel_mult):
            out_ch = int(mult * base_channels)
            for _ in range(num_res_blocks):
                block = nn.ModuleList([CCDMResBlock(ch, emb_ch, out_ch)])
                ch = out_ch
                if ds in attn_ds:
                    block.append(CCDMAttentionBlock(ch, num_head_channels))
                self.down_blocks.append(block)
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                input_block_chans.append(ch)
                ds *= 2

        self.mid_block1 = CCDMResBlock(ch, emb_ch, ch)
        self.mid_attn = CCDMAttentionBlock(ch, num_head_channels)
        self.mid_block2 = CCDMResBlock(ch, emb_ch, ch)

        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = int(mult * base_channels)
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                block = nn.ModuleList([CCDMResBlock(ch + ich, emb_ch, out_ch)])
                ch = out_ch
                if ds in attn_ds:
                    block.append(CCDMAttentionBlock(ch, num_head_channels))
                if level > 0 and i == num_res_blocks:
                    block.append(nn.Conv2d(ch, ch, 3, padding=1))
                    ds //= 2
                self.up_blocks.append(block)

        self.out = nn.Sequential(
            nn.GroupNorm(32 if ch >= 32 else 8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    @staticmethod
    def _run_stage(block: nn.ModuleList, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if len(block) == 1 and isinstance(block[0], nn.Conv2d):
            return block[0](h)
        h = block[0](h, emb)
        for layer in block[1:]:
            if isinstance(layer, CCDMAttentionBlock):
                h = layer(h)
            elif isinstance(layer, nn.Conv2d):
                b, c, hh, ww = h.shape
                h_up = h[:, :, :, None, :, None].expand(b, c, hh, 2, ww, 2).reshape(b, c, hh * 2, ww * 2)
                h = layer(h_up)
        return h

    def forward(self, xt: torch.Tensor, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(ccdm_timestep_embedding(timesteps, self.base_channels))
        h = self.input_conv(torch.cat([xt, image], dim=1))
        hs = [h]
        for block in self.down_blocks:
            h = self._run_stage(block, h, emb)
            hs.append(h)
        h = self.mid_block2(self.mid_attn(self.mid_block1(h, emb)), emb)
        for block in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = self._run_stage(block, h, emb)
        return self.out(h)


class CCDM(nn.Module):
    """Conditional Categorical Diffusion Model (Zbinden et al., ICCV 2023)."""
    is_diffusion = True
    is_categorical = True

    def __init__(self, num_classes: int = 4, time_steps: int = 250, sampling_steps: int = 10, s: float = 0.008):
        super().__init__()
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.sampling_steps = sampling_steps

        # Hoogeboom et al. (2021) cosine schedule (ddpm/models/diffusion_denoising.py lines 341-355)
        t = torch.arange(0, time_steps, dtype=torch.float64)
        cumalphas = torch.cos(((t / time_steps + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2
        func = lambda u: math.cos((u + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
        betas = [min(1.0 - func((i + 1) / time_steps) / func(i / time_steps), 0.999) for i in range(time_steps)]
        betas_t = torch.tensor(betas, dtype=torch.float32)
        alphas_t = 1.0 - betas_t
        self.register_buffer('betas', betas_t)
        self.register_buffer('alphas', alphas_t)
        self.register_buffer('cumalphas', cumalphas.float())

        self.unet = CCDMUNet(in_channels=num_classes + 3, base_channels=64, out_channels=num_classes)

    def q_xt_given_x0_probs(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ca = self.cumalphas[t - 1][:, None, None, None]
        return ca * x0 + (1.0 - ca) / self.num_classes

    def theta_post(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        idx = t - 1
        alphas_t = self.alphas[idx][:, None, None, None].clone()
        cumalphas_tm1 = self.cumalphas[(idx - 1).clamp(min=0)][:, None, None, None].clone()
        is_t1 = (idx == 0)[:, None, None, None]
        alphas_t = torch.where(is_t1, torch.zeros_like(alphas_t), alphas_t)
        cumalphas_tm1 = torch.where(is_t1, torch.ones_like(cumalphas_tm1), cumalphas_tm1)
        theta = (alphas_t * xt + (1.0 - alphas_t) / self.num_classes) * (
            cumalphas_tm1 * x0 + (1.0 - cumalphas_tm1) / self.num_classes
        )
        return theta / theta.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def theta_post_prob(self, xt: torch.Tensor, theta_x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        idx = t - 1
        alphas_t = self.alphas[idx][:, None, None, None].clone()
        cumalphas_tm1 = self.cumalphas[(idx - 1).clamp(min=0)][:, None, None, None, None].clone()
        is_t1 = (idx == 0)
        alphas_t = torch.where(is_t1[:, None, None, None], torch.zeros_like(alphas_t), alphas_t)
        cumalphas_tm1 = torch.where(is_t1[:, None, None, None, None], torch.ones_like(cumalphas_tm1), cumalphas_tm1)

        x0 = torch.eye(self.num_classes, device=xt.device, dtype=xt.dtype)[None, :, :, None, None]
        theta_xt_xtm1 = alphas_t * xt + (1.0 - alphas_t) / self.num_classes
        theta_xtm1_x0 = cumalphas_tm1 * x0 + (1.0 - cumalphas_tm1) / self.num_classes
        aux = theta_xt_xtm1[:, :, None] * theta_xtm1_x0
        theta_xtm1_xtx0 = aux / aux.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.einsum('bcdhw,bdhw->bchw', theta_xtm1_xtx0, theta_x0)

    def theta_post_prob_jump(self, xt: torch.Tensor, theta_x0: torch.Tensor, t_cur: int, t_next: int) -> torch.Tensor:
        """Exact categorical posterior q(x_{t_next} | x_{t_cur}, x0_pred) for stride step t_cur -> t_next."""
        ca_cur = self.cumalphas[t_cur - 1]
        ca_next = self.cumalphas[t_next - 1]
        alpha_jump = (ca_cur / ca_next.clamp_min(1e-12)).clamp(0.0, 1.0)
        x0 = torch.eye(self.num_classes, device=xt.device, dtype=xt.dtype)[None, :, :, None, None]
        theta_xt_xs = alpha_jump * xt + (1.0 - alpha_jump) / self.num_classes
        theta_xs_x0 = ca_next * x0 + (1.0 - ca_next) / self.num_classes
        aux = theta_xt_xs[:, :, None] * theta_xs_x0
        theta_xs_xtx0 = aux / aux.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.einsum('bcdhw,bdhw->bchw', theta_xs_xtx0, theta_x0)

    def forward_train(self, x: torch.Tensor, mask: torch.Tensor):
        b = x.shape[0]
        x0 = F.one_hot(mask, self.num_classes).permute(0, 3, 1, 2).float()
        t = torch.randint(1, self.time_steps + 1, size=(b,), device=x.device)
        xt = sample_categorical_det(self.q_xt_given_x0_probs(x0, t))

        logits = self.unet(xt, x, t.float())
        x0_pred = logits.softmax(dim=1)

        prob_gt = self.theta_post(xt, x0, t)
        prob_pred = self.theta_post_prob(xt, x0_pred, t)
        loss_kl = (prob_gt * (prob_gt.clamp_min(1e-12).log() - prob_pred.clamp_min(1e-12).log())).sum(dim=1).mean()

        sem_loss, terms = standard_sem_loss(logits, mask)
        total = loss_kl + sem_loss
        terms['loss_kl'] = loss_kl
        terms['loss_total'] = total
        return logits, terms

    @torch.no_grad()
    def infer_windows(self, x: torch.Tensor, u_field: torch.Tensor = None, steps: int = None):
        steps = self.sampling_steps if steps is None else steps
        b, _, h, w = x.shape
        t_values = [int(round(v)) for v in np.linspace(self.time_steps, 1, steps)]
        u0 = None if u_field is None else u_field[:, 0]
        xt = sample_categorical_det(torch.full((b, self.num_classes, h, w), 1.0 / self.num_classes, device=x.device, dtype=x.dtype), u0)

        logits = None
        for idx, t_cur in enumerate(t_values):
            t_tensor = torch.full((b,), t_cur, device=x.device, dtype=torch.long)
            logits = self.unet(xt, x, t_tensor.float())
            x0_pred = logits.softmax(dim=1)
            if idx < len(t_values) - 1:
                t_next = t_values[idx + 1]
                probs = self.theta_post_prob_jump(xt, x0_pred, t_cur, t_next)
                ui = None if u_field is None else u_field[:, idx + 1]
                xt = sample_categorical_det(probs, ui)
        return logits, b * len(t_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.infer_windows(x)
        return logits

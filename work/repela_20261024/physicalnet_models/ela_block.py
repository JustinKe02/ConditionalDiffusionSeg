"""
Efficient Linear Attention Block (ELA Block)
Inspired by EfficientViT (CVPR 2024).

Key features:
  - Multi-scale linear attention with O(n) complexity
  - Q pooling at multiple scales to capture multi-scale features
  - Depthwise Conv for local feature extraction
  - Lightweight FFN with DWConv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLinearAttention(nn.Module):
    """Multi-scale Linear Attention module.
    
    Instead of O(n^2) standard self-attention, uses kernel trick for O(n):
        Attn(Q, K, V) = phi(Q) * (phi(K)^T * V)
    
    Multi-scale Q pooling captures features at different scales,
    crucial for MoS2 images with objects spanning 70 to 2.6M pixels.
    
    All core computations forced to float32 for AMP stability.
    """

    def __init__(self, dim, num_heads=4, qk_scale=None, attn_drop=0.,
                 scales=(1, 2, 4)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.scales = scales

        # Q, K, V projections
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

        # Learnable scale weights for multi-scale fusion
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

    def _linear_attention(self, q, k, v):
        """Compute linear attention using kernel trick. Always in float32.
        
        phi(x) = elu(x) + 1 (ensures non-negative)
        Attn = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ sum(phi(K)))
        """
        # Force float32 for numerical stability under AMP
        q = q.float()
        k = k.float()
        v = v.float()

        # Scale down Q and K to prevent large accumulations
        q = q * self.scale
        k = k * self.scale

        q = F.elu(q) + 1  # non-negative feature map
        k = F.elu(k) + 1

        # (phi(K)^T @ V): [B, H, D, D]
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        # phi(Q) @ (phi(K)^T @ V): [B, H, N, D]
        qkv = torch.einsum('bhnd,bhde->bhne', q, kv)
        # Normalization: phi(Q) @ sum(phi(K))
        k_sum = k.sum(dim=2, keepdim=True)  # [B, H, 1, D]
        normalizer = torch.einsum('bhnd,bhkd->bhn', q, k_sum).unsqueeze(-1)
        normalizer = normalizer.clamp(min=1e-6)

        out = qkv / normalizer
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Multi-scale attention
        scale_weights = F.softmax(self.scale_weights, dim=0)

        # Full resolution attention (always computed)
        out = scale_weights[0] * self._linear_attention(q, k, v)

        # Pooled Q scales for larger receptive fields
        for i, s in enumerate(self.scales[1:], 1):
            q_reshaped = q.reshape(B, self.num_heads, H, W, self.head_dim)
            # Pool along spatial dims
            q_pooled = q_reshaped.permute(0, 1, 4, 2, 3)  # [B, H, D, h, w]
            q_pooled = q_pooled.reshape(B * self.num_heads * self.head_dim, 1, H, W)
            q_pooled = F.adaptive_avg_pool2d(q_pooled, (max(1, H // s), max(1, W // s)))
            pH, pW = q_pooled.shape[2], q_pooled.shape[3]
            q_pooled = q_pooled.reshape(B, self.num_heads, self.head_dim, pH, pW)
            q_pooled = q_pooled.permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, pH * pW, self.head_dim)

            # Attention with pooled Q
            attn_out = self._linear_attention(q_pooled, k, v)

            # Upsample back
            attn_out = attn_out.reshape(B, self.num_heads, pH, pW, self.head_dim)
            attn_out = attn_out.permute(0, 1, 4, 2, 3)  # [B, H, D, pH, pW]
            attn_out = attn_out.reshape(B * self.num_heads * self.head_dim, 1, pH, pW)
            attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)
            attn_out = attn_out.reshape(B, self.num_heads, self.head_dim, H, W)
            attn_out = attn_out.permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, N, self.head_dim)

            out = out + scale_weights[i] * attn_out

        out = self.attn_drop(out)

        # Reshape back to spatial
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        out = self.proj(out)

        return out


class LightweightFFN(nn.Module):
    """Lightweight Feed-Forward Network with Depthwise Conv.
    
    Structure: 1x1 Conv -> DWConv 3x3 -> GELU -> 1x1 Conv
    The DWConv provides local context within the FFN.
    """

    def __init__(self, dim, expand_ratio=2, drop=0.):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dw = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1,
                            groups=hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ELABlock(nn.Module):
    """Efficient Linear Attention Block.
    
    Structure:
        x -> LayerNorm -> DWConv (local) -> Multi-scale Linear Attn -> Residual
          -> LayerNorm -> Lightweight FFN -> Residual
    """

    def __init__(self, dim, num_heads=4, expand_ratio=2, drop=0.,
                 attn_drop=0., scales=(1, 2, 4)):
        super().__init__()
        # Pre-norm
        self.norm1 = nn.GroupNorm(1, dim)  # equivalent to LayerNorm for conv
        self.norm2 = nn.GroupNorm(1, dim)

        # Local feature extraction
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # Multi-scale linear attention
        self.attn = MultiScaleLinearAttention(
            dim, num_heads=num_heads, attn_drop=attn_drop, scales=scales
        )

        # FFN
        self.ffn = LightweightFFN(dim, expand_ratio=expand_ratio, drop=drop)

        # Layer scale (learnable, from CaiT/ConvNeXt)
        self.layer_scale_1 = nn.Parameter(1e-5 * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(1e-5 * torch.ones(dim))

    def forward(self, x):
        # Attention branch
        shortcut = x
        x = self.norm1(x)
        x = self.local_conv(x)
        x = self.attn(x)
        x = shortcut + self.layer_scale_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * x

        # FFN branch
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut + self.layer_scale_2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * x

        return x


class ELAStage(nn.Module):
    """A stage consisting of a downsample layer + multiple ELA blocks."""

    def __init__(self, in_channels, out_channels, num_blocks, num_heads=4,
                 expand_ratio=2, drop=0., attn_drop=0., scales=(1, 2, 4)):
        super().__init__()
        # Downsample: Conv 2x2 stride 2
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # ELA blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ELABlock(
                out_channels, num_heads=num_heads,
                expand_ratio=expand_ratio, drop=drop,
                attn_drop=attn_drop, scales=scales
            ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x

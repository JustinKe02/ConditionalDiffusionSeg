"""
Dynamic Weighted Multi-scale Fusion Decoder (DW-MFF Decoder)
Inspired by BiFPN dynamic weighting + EMCAD (CVPR 2024) multi-scale attention.

Key features:
  - Learnable dynamic weights for multi-scale feature fusion
  - Lightweight depthwise separable convolution for refinement
  - Boundary-aware enhancement at the final stage
  - Deep supervision with auxiliary heads (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicWeightedFusion(nn.Module):
    """Dynamically weighted fusion of two feature maps.
    
    Instead of simple addition or concatenation, uses learnable weights
    with fast normalized fusion (from BiFPN):
        out = (w1 * F1 + w2 * F2) / (w1 + w2 + eps)
    
    This allows the network to learn the relative importance of features
    at each scale, critical for MoS2 images where object sizes span
    from 70 to 2.6M pixels.
    """

    def __init__(self, channels, num_inputs=2):
        super().__init__()
        self.num_inputs = num_inputs
        # Learnable weights (initialized to equal contribution)
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.eps = 1e-4

        # Lightweight refinement after fusion
        self.refine = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            # Pointwise conv
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, features):
        """
        Args:
            features: list of tensors, each [B, C, H, W]
                      All must have the same spatial size.
        """
        assert len(features) == self.num_inputs

        # Fast normalized fusion weights (always positive via ReLU)
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)

        # Weighted sum
        out = sum(w[i] * features[i] for i in range(self.num_inputs))
        out = self.refine(out)
        return out


class BoundaryEnhancement(nn.Module):
    """Boundary Enhancement Module.
    
    Designed specifically for MoS2 material images where boundaries
    between different layer thicknesses are often blurry and subtle.
    
    Uses gradient-based edge detection combined with learned features.
    """

    def __init__(self, channels):
        super().__init__()
        # Edge detection branch (Sobel-like learnable filters)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # Channel attention for boundary features
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Edge-aware attention
        edge = self.edge_conv(x)
        # Channel attention
        ca = self.channel_attn(x)
        # Enhance boundaries
        out = x + x * edge * ca
        return out


class DWMFFDecoder(nn.Module):
    """Dynamic Weighted Multi-scale Feature Fusion Decoder.
    
    Takes feature maps from 4 encoder stages and progressively
    fuses them from deep to shallow using dynamic weights.
    Supports deep supervision with auxiliary classification heads.
    
    Architecture:
        F4 (1/32) ──────────────────────┐
        F3 (1/16) ──────────────┐       │
                                └───> DWFuse -> up -> ┐  (aux_head_3)
        F2 (1/8)  ──────┐                              │
                         └─────────────────> DWFuse -> up -> ┐  (aux_head_2)
        F1 (1/4)  ──┐                                      │
                     └─────────────────────> DWFuse + BoundaryEnhance -> output
    """

    def __init__(self, in_channels_list, decoder_channels=128,
                 num_classes=4, deep_supervision=False):
        """
        Args:
            in_channels_list: [C1, C2, C3, C4] channel counts from encoder stages
            decoder_channels: unified channel count in the decoder
            num_classes: number of segmentation classes
            deep_supervision: if True, add auxiliary classification heads
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        c1, c2, c3, c4 = in_channels_list

        # Channel alignment (project all features to decoder_channels)
        self.align4 = nn.Sequential(
            nn.Conv2d(c4, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels)
        )
        self.align3 = nn.Sequential(
            nn.Conv2d(c3, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels)
        )
        self.align2 = nn.Sequential(
            nn.Conv2d(c2, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels)
        )
        self.align1 = nn.Sequential(
            nn.Conv2d(c1, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels)
        )

        # Fusion modules (deep to shallow)
        self.fuse_43 = DynamicWeightedFusion(decoder_channels, num_inputs=2)
        self.fuse_32 = DynamicWeightedFusion(decoder_channels, num_inputs=2)
        self.fuse_21 = DynamicWeightedFusion(decoder_channels, num_inputs=2)

        # Boundary enhancement at the finest scale
        self.boundary = BoundaryEnhancement(decoder_channels)

        # Main segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, num_classes, 1)
        )

        # Auxiliary heads for deep supervision (only used during training)
        if deep_supervision:
            self.aux_head_3 = nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels // 2, 1, bias=False),
                nn.BatchNorm2d(decoder_channels // 2),
                nn.GELU(),
                nn.Conv2d(decoder_channels // 2, num_classes, 1)
            )
            self.aux_head_2 = nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels // 2, 1, bias=False),
                nn.BatchNorm2d(decoder_channels // 2),
                nn.GELU(),
                nn.Conv2d(decoder_channels // 2, num_classes, 1)
            )

    def forward(self, features):
        """
        Args:
            features: [F1, F2, F3, F4] from encoder stages
                F1: [B, C1, H/4, W/4]
                F2: [B, C2, H/8, W/8]
                F3: [B, C3, H/16, W/16]
                F4: [B, C4, H/32, W/32]
        
        Returns:
            If deep_supervision and training:
                (main_logits, [aux3_logits, aux2_logits])
            Else:
                main_logits: [B, num_classes, H/4, W/4]
        """
        f1, f2, f3, f4 = features

        # Align channels
        f4 = self.align4(f4)
        f3 = self.align3(f3)
        f2 = self.align2(f2)
        f1 = self.align1(f1)

        # Fuse F4 + F3
        f4_up = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.fuse_43([f3, f4_up])

        # Fuse P3 + F2
        p3_up = F.interpolate(p3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.fuse_32([f2, p3_up])

        # Fuse P2 + F1
        p2_up = F.interpolate(p2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.fuse_21([f1, p2_up])

        # Boundary enhancement
        p1 = self.boundary(p1)

        # Main segmentation output
        out = self.seg_head(p1)

        if self.deep_supervision and self.training:
            aux3 = self.aux_head_3(p3)
            aux2 = self.aux_head_2(p2)
            return out, [aux3, aux2]

        return out

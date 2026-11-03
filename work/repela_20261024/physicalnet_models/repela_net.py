"""
RepELA-Net: Reparameterized Efficient Linear Attention Network
for Lightweight 2D Material Segmentation.

Complete model assembly combining:
  1. Stem (initial convolution + color space enhancement)
  2. RepConv Stages (stages 1-2, local features)
  3. ELA Stages (stages 3-4, global features)
  4. DW-MFF Decoder (dynamic weighted multi-scale fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rep_conv import RepConvStage
from .ela_block import ELAStage
from .decoder import DWMFFDecoder


class ColorSpaceEnhancement(nn.Module):
    """Color Space Enhancement Module.
    
    MoS2 material images have subtle color differences between layers.
    This module extracts the Saturation channel from HSV color space
    and concatenates it with RGB, providing more discriminative color features.
    
    Input: RGB [B, 3, H, W]
    Output: Enhanced [B, 4, H, W] (RGB + S_channel)
    """

    def __init__(self, norm_mean=None, norm_std=None):
        super().__init__()
        # Learnable weight for the saturation channel
        self.s_weight = nn.Parameter(torch.tensor(1.0))
        # Normalization constants for denormalization (default: ImageNet)
        if norm_mean is None:
            norm_mean = [0.485, 0.456, 0.406]
        if norm_std is None:
            norm_std = [0.229, 0.224, 0.225]
        self.register_buffer('mean', torch.tensor(norm_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(norm_std).view(1, 3, 1, 1))

    def rgb_to_saturation(self, rgb_normalized):
        """Extract saturation channel from normalized RGB image.
        
        First denormalizes to [0,1] RGB, then computes:
        Saturation = (Max - Min) / Max, where Max and Min are
        per-pixel max/min across R, G, B channels.
        """
        # Denormalize: x_orig = x * std + mean → [0, 1] range
        rgb_01 = rgb_normalized * self.std + self.mean
        rgb_01 = rgb_01.clamp(0.0, 1.0)
        max_rgb = rgb_01.max(dim=1, keepdim=True)[0]
        min_rgb = rgb_01.min(dim=1, keepdim=True)[0]
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
        return saturation

    def forward(self, x):
        # x: [B, 3, H, W] normalized RGB
        sat = self.rgb_to_saturation(x)  # [B, 1, H, W]
        # Concatenate normalized RGB with weighted saturation
        enhanced = torch.cat([x, self.s_weight * sat], dim=1)
        return enhanced


class ZeroPadChannel(nn.Module):
    """Replace CSE with a zero-filled 4th channel.

    Keeps the Stem input as 4 channels (same architecture) but without
    color space features. This is the default for RepELA-Net (w/o CSE).
    """
    def forward(self, x):
        zeros = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3],
                            device=x.device, dtype=x.dtype)
        return torch.cat([x, zeros], dim=1)


def infer_use_cse(checkpoint, cli_use_cse=False):
    """Infer use_cse from a checkpoint dict.

    Priority:
        1. checkpoint['args']['use_cse'] (new checkpoints)
        2. 'color_enhance.s_weight' in state_dict (old CSE checkpoints)
        3. cli_use_cse fallback (default False)

    Args:
        checkpoint: loaded checkpoint dict (must have 'model' key)
        cli_use_cse: fallback value from CLI

    Returns:
        bool: whether to enable CSE
    """
    # 1. Explicit args in checkpoint
    ckpt_args = checkpoint.get('args', {})
    if isinstance(ckpt_args, dict) and 'use_cse' in ckpt_args:
        return ckpt_args['use_cse']

    # 2. Infer from state_dict keys
    sd = checkpoint.get('model', checkpoint)
    if any('color_enhance.s_weight' in k for k in sd.keys()):
        return True

    # 3. CLI fallback
    return cli_use_cse


class RepELANet(nn.Module):
    """RepELA-Net: Complete lightweight segmentation model.
    
    Architecture overview:
        Input (512x512x3)
        -> ZeroPadChannel / ColorSpaceEnhancement (512x512x4)
        -> Stem Conv (256x256xC1=32)
        -> RepConv Stage 1 (128x128xC1=32)
        -> RepConv Stage 2 (64x64xC2=64)
        -> ELA Stage 3 (32x32xC3=128)
        -> ELA Stage 4 (16x16xC4=256)
        -> DW-MFF Decoder (128x128x4)
        -> Upsample (512x512x4)
    
    Target: <3M params, <2G FLOPs @512x512
    """

    def __init__(self, num_classes=4, channels=(32, 64, 128, 256),
                 num_blocks=(2, 2, 4, 2), num_heads=(0, 0, 4, 8),
                 decoder_channels=128, drop_rate=0.1, deploy=False,
                 deep_supervision=False, use_cse=False,
                 norm_mean=None, norm_std=None):
        """
        Args:
            num_classes: number of segmentation classes (4 for MoS2 dataset)
            channels: channel count at each stage
            num_blocks: number of blocks at each stage
            num_heads: attention heads at each stage (0 = RepConv, >0 = ELA)
            decoder_channels: channel count in decoder
            drop_rate: dropout rate
            deploy: if True, use fused/deployed convolutions
            deep_supervision: if True, add auxiliary classification heads
            use_cse: if True, use ColorSpaceEnhancement (HSV saturation);
                     if False (default), use ZeroPadChannel (zero-filled 4th channel)
            norm_mean: RGB normalization mean for CSE denormalization (default: ImageNet)
            norm_std: RGB normalization std for CSE denormalization (default: ImageNet)
        """
        super().__init__()
        self.num_classes = num_classes
        c1, c2, c3, c4 = channels

        # Input channel expansion: 3ch RGB -> 4ch
        if use_cse:
            self.color_enhance = ColorSpaceEnhancement(norm_mean=norm_mean, norm_std=norm_std)
        else:
            self.color_enhance = ZeroPadChannel()

        # Stem: initial downsampling (4x4 -> 1/4 resolution)
        self.stem = nn.Sequential(
            nn.Conv2d(4, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )

        # Stage 1: RepConv (1/4 -> 1/8)
        self.stage1 = RepConvStage(
            c1, c1, num_blocks=num_blocks[0],
            expand_ratio=2, use_se=True, deploy=deploy
        )

        # Stage 2: RepConv (1/8 -> 1/16)
        self.stage2 = RepConvStage(
            c1, c2, num_blocks=num_blocks[1],
            expand_ratio=2, use_se=True, deploy=deploy
        )

        # Stage 3: ELA (1/16 -> 1/32)
        self.stage3 = ELAStage(
            c2, c3, num_blocks=num_blocks[2],
            num_heads=num_heads[2], expand_ratio=2,
            drop=drop_rate, attn_drop=0., scales=(1, 2, 4)
        )

        # Stage 4: ELA (1/32 -> 1/64)
        self.stage4 = ELAStage(
            c3, c4, num_blocks=num_blocks[3],
            num_heads=num_heads[3], expand_ratio=2,
            drop=drop_rate, attn_drop=0., scales=(1, 2)
        )

        # Decoder
        self.decoder = DWMFFDecoder(
            in_channels_list=[c1, c2, c3, c4],
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            deep_supervision=deep_supervision
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: input image [B, 3, H, W]
        Returns:
            logits: [B, num_classes, H, W]
        """
        input_size = x.shape[2:]

        # Color space enhancement
        x = self.color_enhance(x)  # [B, 4, H, W]

        # Stem
        x = self.stem(x)  # [B, C1, H/2, W/2]

        # Encoder stages
        f1 = self.stage1(x)   # [B, C1, H/4, W/4]
        f2 = self.stage2(f1)  # [B, C2, H/8, W/8]
        f3 = self.stage3(f2)  # [B, C3, H/16, W/16]
        f4 = self.stage4(f3)  # [B, C4, H/32, W/32]

        # Decoder
        decoder_out = self.decoder([f1, f2, f3, f4])

        if isinstance(decoder_out, tuple):
            # Deep supervision: (main_logits, [aux3, aux2])
            out, aux_list = decoder_out
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
            aux_list = [F.interpolate(a, size=input_size, mode='bilinear', align_corners=False)
                        for a in aux_list]
            return out, aux_list
        else:
            out = decoder_out
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
            return out

    def switch_to_deploy(self):
        """Switch all reparameterizable modules to deploy/inference mode."""
        self.stage1.switch_to_deploy()
        self.stage2.switch_to_deploy()
        print("[RepELA-Net] Switched to deploy mode (reparameterized)")

    def get_params_flops(self, input_size=(512, 512)):
        """Compute parameter count and FLOPs."""
        from thop import profile, clever_format
        dummy = torch.randn(1, 3, *input_size).to(next(self.parameters()).device)
        flops, params = profile(self, inputs=(dummy,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.2f")
        return params, flops, params_str, flops_str


def repela_net_tiny(num_classes=4, deploy=False, deep_supervision=False, use_cse=False, **kwargs):
    """RepELA-Net-Tiny: ~1.13M params."""
    return RepELANet(
        num_classes=num_classes,
        channels=(24, 48, 96, 192),
        num_blocks=(2, 2, 3, 2),
        num_heads=(0, 0, 4, 8),
        decoder_channels=96,
        drop_rate=0.05,
        deploy=deploy,
        deep_supervision=deep_supervision,
        use_cse=use_cse,
        **kwargs
    )


def repela_net_small(num_classes=4, deploy=False, deep_supervision=False, use_cse=False, **kwargs):
    """RepELA-Net-Small: ~2.12M params (recommended for MoS2)."""
    return RepELANet(
        num_classes=num_classes,
        channels=(32, 64, 128, 256),
        num_blocks=(2, 2, 4, 2),
        num_heads=(0, 0, 4, 8),
        decoder_channels=128,
        drop_rate=0.1,
        deploy=deploy,
        deep_supervision=deep_supervision,
        use_cse=use_cse,
        **kwargs
    )


def repela_net_base(num_classes=4, deploy=False, deep_supervision=False, use_cse=False, **kwargs):
    """RepELA-Net-Base: ~5.34M params (higher accuracy)."""
    return RepELANet(
        num_classes=num_classes,
        channels=(48, 96, 192, 384),
        num_blocks=(2, 2, 6, 2),
        num_heads=(0, 0, 6, 12),
        decoder_channels=192,
        drop_rate=0.1,
        deploy=deploy,
        deep_supervision=deep_supervision,
        use_cse=use_cse,
        **kwargs
    )

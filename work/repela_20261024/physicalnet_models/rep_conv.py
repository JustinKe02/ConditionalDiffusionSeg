"""
Reparameterized Convolution Block (RepConv Block)
Inspired by RepViT (CVPR 2024) and UniRepLKNet (CVPR 2024).

Training: Multi-branch topology (3x3 Conv + 1x1 Conv + Identity + BN)
Inference: Merged into a single 3x3 Conv (zero extra cost)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RepConvBN(nn.Module):
    """Reparameterizable Conv-BN block.
    
    During training, maintains parallel branches:
      - 3x3 depthwise conv + BN
      - 1x1 conv + BN  
      - Identity + BN (when in_channels == out_channels)
    
    During inference, all branches are fused into a single 3x3 conv.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        if deploy:
            # Inference mode: single fused conv
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=True
            )
        else:
            # Training mode: multi-branch
            # Branch 1: 3x3 conv + BN
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            # Branch 2: 1x1 conv + BN
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, padding=0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            # Branch 3: Identity + BN (only when dimensions match)
            if in_channels == out_channels and stride == 1:
                self.identity = nn.BatchNorm2d(out_channels)
            else:
                self.identity = None

    def forward(self, x):
        if self.deploy:
            return self.fused_conv(x)

        out = self.conv3x3(x) + self.conv1x1(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return out

    def _fuse_bn(self, conv, bn):
        """Fuse Conv and BatchNorm into a single Conv with bias."""
        kernel = conv.weight
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(var + eps)
        fused_weight = kernel * (gamma / std).reshape(-1, 1, 1, 1)
        fused_bias = beta - mean * gamma / std
        return fused_weight, fused_bias

    def _pad_1x1_to_3x3(self, kernel):
        """Pad a 1x1 kernel to 3x3."""
        if self.kernel_size == 3:
            return F.pad(kernel, [1, 1, 1, 1])
        else:
            raise ValueError(f"Unsupported kernel size: {self.kernel_size}")

    def _get_identity_kernel_bias(self):
        """Get equivalent kernel and bias for identity + BN branch."""
        assert self.identity is not None
        kernel = torch.zeros(
            self.out_channels, self.in_channels // self.groups,
            self.kernel_size, self.kernel_size,
            device=self.identity.weight.device
        )
        for i in range(self.out_channels):
            kernel[i, i % (self.in_channels // self.groups),
                   self.kernel_size // 2, self.kernel_size // 2] = 1

        bn = self.identity
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        std = torch.sqrt(var + eps)

        fused_weight = kernel * (gamma / std).reshape(-1, 1, 1, 1)
        fused_bias = beta - mean * gamma / std
        return fused_weight, fused_bias

    def switch_to_deploy(self):
        """Convert from training mode to deploy mode (fuse all branches)."""
        if self.deploy:
            return

        # Fuse 3x3 branch
        w3, b3 = self._fuse_bn(self.conv3x3[0], self.conv3x3[1])

        # Fuse 1x1 branch and pad to 3x3
        w1, b1 = self._fuse_bn(self.conv1x1[0], self.conv1x1[1])
        w1 = self._pad_1x1_to_3x3(w1)

        # Merge all branches
        fused_weight = w3 + w1
        fused_bias = b3 + b1

        if self.identity is not None:
            wi, bi = self._get_identity_kernel_bias()
            fused_weight = fused_weight + wi
            fused_bias = fused_bias + bi

        # Create fused conv
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding,
            groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = fused_weight
        self.fused_conv.bias.data = fused_bias

        # Remove training branches
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
        if hasattr(self, 'identity') and self.identity is not None:
            self.__delattr__('identity')

        self.deploy = True


class RepConvBlock(nn.Module):
    """Complete Reparameterizable Convolution Block.
    
    Structure: RepConvBN (depthwise) -> Act -> RepConvBN (pointwise) -> Act -> SE
    
    This is inspired by MobileNetV2's inverted residual but with
    reparameterizable convolutions for enhanced training.
    """

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2,
                 use_se=True, deploy=False):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        mid_channels = int(in_channels * expand_ratio)

        # Expansion: 1x1 pointwise conv
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU()
        )

        # Depthwise: Reparameterizable 3x3 depthwise conv
        self.dw_rep = RepConvBN(
            mid_channels, mid_channels, kernel_size=3,
            stride=stride, padding=1, groups=mid_channels, deploy=deploy
        )
        self.dw_act = nn.GELU()

        # Squeeze-and-Excitation
        if use_se:
            se_channels = max(1, in_channels // 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(mid_channels, se_channels, 1),
                nn.GELU(),
                nn.Conv2d(se_channels, mid_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        # Projection: 1x1 pointwise conv
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.expand(x)
        out = self.dw_act(self.dw_rep(out))

        if self.se is not None:
            out = out * self.se(out)

        out = self.project(out)

        if self.use_residual:
            out = out + identity
        return out

    def switch_to_deploy(self):
        """Switch RepConvBN to deploy mode."""
        self.dw_rep.switch_to_deploy()


class RepConvStage(nn.Module):
    """A stage consisting of multiple RepConvBlocks.
    
    First block handles downsampling (stride=2), rest maintain resolution.
    """

    def __init__(self, in_channels, out_channels, num_blocks, expand_ratio=2,
                 use_se=True, deploy=False):
        super().__init__()
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            in_ch = in_channels if i == 0 else out_channels
            layers.append(RepConvBlock(
                in_ch, out_channels, stride=stride,
                expand_ratio=expand_ratio, use_se=use_se, deploy=deploy
            ))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)

    def switch_to_deploy(self):
        for block in self.blocks:
            block.switch_to_deploy()

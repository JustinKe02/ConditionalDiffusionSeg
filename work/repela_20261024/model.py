"""RepELA-Net-Small wrapper for the unified Chapter-4 protocol (2026-10-24).

The architecture comes unmodified from physicalnet_models/ (byte-identical copies of PhysicalNet/models).
Two runtime substitutions keep CUDA backward passes deterministic under torch.use_deterministic_algorithms(True):
  - F.interpolate(mode='bilinear', align_corners=False) and F.adaptive_avg_pool2d inside the copied modules are
    routed to the deterministic matrix forms in baselines_20261022.models (same operators as the 9 baselines);
  - nn.AdaptiveAvgPool2d(1) modules (SE and boundary channel attention) are replaced by a spatial mean.
Both are forward-equivalent up to float rounding; engineering verification measures the difference.
`native_ops()` disables the routing (used for the reference forward and for inference benchmarking).
"""
import contextlib
import torch
from torch import nn
from torch.nn import functional as F

from baselines_20261022.models import det_interpolate, det_adaptive_avg_pool2d, standard_sem_loss
from .physicalnet_models import repela_net, ela_block, decoder
from .physicalnet_models.repela_net import repela_net_small

_DET = {'on': True}


class _RoutedFunctional:
    """Stands in for torch.nn.functional inside the copied modules; only two calls are rerouted."""

    def __getattr__(self, name):
        return getattr(F, name)

    @staticmethod
    def interpolate(x, size=None, scale_factor=None, mode='nearest', align_corners=None, **kw):
        if _DET['on'] and mode == 'bilinear' and not align_corners and not kw:
            return det_interpolate(x, size=size, scale_factor=scale_factor)
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, **kw)

    @staticmethod
    def adaptive_avg_pool2d(x, output_size):
        if _DET['on']:
            return det_adaptive_avg_pool2d(x, output_size)
        return F.adaptive_avg_pool2d(x, output_size)


for _mod in (repela_net, ela_block, decoder):
    _mod.F = _RoutedFunctional()


@contextlib.contextmanager
def native_ops():
    prev = _DET['on']
    _DET['on'] = False
    try:
        yield
    finally:
        _DET['on'] = prev


class GlobalMean(nn.Module):
    """nn.AdaptiveAvgPool2d(1) with a deterministic backward."""

    def forward(self, x):
        return x.mean(dim=(-2, -1), keepdim=True)


def _replace_global_pools(module: nn.Module) -> int:
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.AdaptiveAvgPool2d) and child.output_size in (1, (1, 1)):
            setattr(module, name, GlobalMean())
            n += 1
        else:
            n += _replace_global_pools(child)
    return n


class RepELASmall(nn.Module):
    """RepELA-Net-Small w/o CSE, no deep supervision (the published-paper configuration)."""

    def __init__(self, num_classes: int = 4, deterministic_pools: bool = True):
        super().__init__()
        self.net = repela_net_small(num_classes=num_classes, deploy=False, deep_supervision=False, use_cse=False)
        self.replaced_pools = _replace_global_pools(self.net) if deterministic_pools else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_train(self, x: torch.Tensor, mask: torch.Tensor):
        logits = self.forward(x)
        _, terms = standard_sem_loss(logits, mask)
        return logits, terms

    def switch_to_deploy(self):
        """Fuse the RepConv branches (stages 1-2) into single 3x3 convolutions, in place."""
        self.net.switch_to_deploy()
        return self

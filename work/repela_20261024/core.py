"""Registers RepELA-Net-Small in the baseline training framework without editing it (2026-10-24).

baselines_20261022.core.train/evaluate/predict are reused as-is: the method is added to its METHODS table and
build_model is wrapped at import time, so the unified protocol (data, augmentation, 100 epochs / 3000 updates,
AdamW, PowerEMA, 512/384 windows, metrics) is literally the same code path as the 9 comparison methods.
"""
from pathlib import Path

from baselines_20261022 import core as B
from .model import RepELASmall

METHOD = 'repela_small'
ROOT = B.ROOT
BATCH = ROOT / 'outputs/launches/p1_ch4_extension_20261024'
RUN = ROOT / 'outputs/P1/repela/seed_42/p1_repela_20261024_repela_small'
TEST_OUT = ROOT / 'outputs/inference/repela_test_20261024'
SOURCE_DIR = Path('/root/autodl-tmp/PhysicalNet/models')
COPY_DIR = Path(__file__).resolve().parent / 'physicalnet_models'
SOURCE_FILES = ('repela_net.py', 'rep_conv.py', 'ela_block.py', 'decoder.py')

B.METHODS.setdefault(METHOD, dict(display='RepELA-Net-Small', is_diffusion=False, nfe_per_window=1, grad_accum_steps=1))
_build_model_baselines = B.build_model


def build_model(c: dict):
    if c['method'] == METHOD:
        return RepELASmall(num_classes=4)
    return _build_model_baselines(c)


B.build_model = build_model


def config() -> dict:
    c = B.config(METHOD)
    c.update(
        schema='repela_v1',
        variant='small',
        use_cse=False,
        deep_supervision=False,
        loss='0.5 CE + 0.5 foreground Dice (unified protocol, same as the 9 baselines)',
        architecture_source='PhysicalNet/models, byte-identical copies in work/repela_20261024/physicalnet_models',
        deterministic_substitutions='bilinear interpolate / adaptive_avg_pool2d -> matrix forms; AdaptiveAvgPool2d(1) -> spatial mean',
    )
    return c

#!/usr/bin/env python3
"""CPU-only Conv/Linear MAC counts for one new-platform inference window.

No training, optimizer, dataset access, checkpoint loading, CUDA call or speed
claim. Hooks count executed operations, including the actual cached five-step
S1/S2 paths. One multiply-accumulate is one MAC, not one FLOP.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from torch import nn

from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.static_semantic import StaticSemanticModel, sample_cached

OUT = ROOT / 'outputs/launches/p1_static_semantic_20261020/complexity.json'


def profile(mode, side):
    model = StaticSemanticModel(mode, seed=42).float().eval()
    image = torch.zeros(1, 3, side, side, dtype=torch.float32)
    noise = torch.randn(1, 4, side, side, generator=torch.Generator().manual_seed(42))
    modules = {}
    handles = []

    def hook(name, module, inputs, output):
        if isinstance(module, nn.Conv2d):
            macs = output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            kind = 'Conv2d'
        else:
            macs = output.numel() * module.in_features
            kind = 'Linear'
        record = modules.setdefault(name, dict(kind=kind, calls=0, macs=0))
        record['calls'] += 1
        record['macs'] += int(macs)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(
                lambda m, a, b, name=name: hook(name, m, a, b)
            ))
    try:
        with torch.inference_mode():
            if mode == 'S2':
                output = sample_cached(model, GaussianDiffusion(1000, 'cpu'), image, noise, steps=5)
            else:
                output = model(image)
    finally:
        for handle in handles:
            handle.remove()
    groups = defaultdict(int)
    kinds = defaultdict(int)
    for name, record in modules.items():
        groups[name.split('.')[0]] += record['macs']
        kinds[record['kind']] += record['macs']
    total = sum(kinds.values())
    return dict(
        mode=mode, input_shape=[1, 3, side, side], dtype='torch.float32', device='cpu',
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        output_shape=list(output.shape), total_conv_linear_macs=total,
        gmacs=total / 1e9, macs_by_module_family=dict(groups), macs_by_operator=dict(kinds),
        executed_conv_linear_calls=sum(x['calls'] for x in modules.values()),
        rgb_encoder_calls=modules['rgb_encoder.input']['calls'],
        state_encoder_calls=modules.get('state_encoder.input', {}).get('calls', 0),
        head_calls=modules['head.2']['calls'],
        inference_steps=1 if mode == 'S0' else 5,
        execution='single direct forward' if mode == 'S0' else (
            'five connected deterministic refinements; RGB encoded once' if mode == 'S1' else
            'cached five-step eta=0 DDIM; RGB encoded once'),
        per_layer_counts=modules,
    )


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = []
    for side in [32, 512]:
        for mode in ['S0', 'S1', 'S2']:
            row = profile(mode, side)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ['mode', 'input_shape', 'parameters', 'gmacs', 'rgb_encoder_calls', 'state_encoder_calls']}), flush=True)
    payload = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Analytical executed Conv2d/Linear MAC count via CPU forward hooks; no training or latency benchmark',
        cpu_threads=2, cpu_interop_threads=1, cuda_visible_devices='', cuda_used=False,
        multiply_accumulate_definition='one scalar multiplication accumulated into an output is one MAC; do not label as FLOPs',
        conv_formula='output.numel() * (in_channels/groups) * kernel_height * kernel_width',
        linear_formula='output.numel() * in_features',
        exclusions=['bias additions', 'GroupNorm', 'SiLU and other activations', 'interpolation',
                    'softmax', 'DDIM arithmetic', 'sinusoidal embedding arithmetic',
                    'concatenation and additive feature fusion', 'memory and transfer operations',
                    'I/O, probability overlap fusion and metrics'],
        initialization='seed42; fresh weights; no checkpoints or data accessed',
        rows=rows,
        full_mos2_35_window_512_conv_linear_gmacs={
            row['mode']: row['gmacs'] * 35 for row in rows if row['input_shape'][-1] == 512
        },
        full_image_note='35 native 512 windows for 2560x1922 at stride384. Window batches do not change sample-level MACs. All excluded operations remain excluded.',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), ROOT / 'src/conditional_diffusion/static_semantic.py',
                                 ROOT / 'src/conditional_diffusion/diffusion.py']},
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(OUT)
    print(str(OUT), flush=True)


if __name__ == '__main__':
    main()

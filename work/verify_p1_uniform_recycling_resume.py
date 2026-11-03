"""Resume equivalence through the registered P1 recycling path; no validation."""
import os
import sys
import json
import time
import argparse
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.trainer import train


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads((ROOT / 'configs/pilot_uniform_recycling.json').read_text())
    c.update(base_channels=8, crop_size=32, stride=32, workers=0, preview_count=0, log_every=1)
    base = ROOT / 'outputs/integration' / ('p1_uniform_recycling_resume_' + str(time.time_ns()))
    full, resumed = base / 'full', base / 'resume'
    for run, stop in [(full, 2), (resumed, 1)]:
        run.mkdir(parents=True)
        train(dict(c), ROOT, run, stop_after_epoch=stop)
    train(dict(c), ROOT, resumed, resume=resumed / 'checkpoints/last.pt', stop_after_epoch=2)
    a, b = [torch.load(run / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
            for run in [full, resumed]]
    checks = {key: equal(a[key], b[key]) for key in ['model', 'optimizer', 'scheduler', 'rng']}
    assert all(checks.values()), checks
    assert a['epoch'] == b['epoch'] == 2 and a['global_step'] == b['global_step'] == 120
    args.output.write_text(json.dumps(dict(status='passed', bitwise_equal=checks, global_step=120,
                                          scope='Registered uniform-timestep P1 recycling, full120 train split, base8/crop32, two controlled epochs of40 schedule, no val/test',
                                          runs=[str(full), str(resumed)]), indent=2))
    print('P1_UNIFORM_RECYCLING_RESUME_PASSED', flush=True)


if __name__ == '__main__':
    main()

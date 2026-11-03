"""One synthetic optimizer step per method, at planned crop/batch; not an experiment."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_diffusion.model import make_model
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.objectives import losses
from conditional_diffusion.runtime import write_json, seed_all


def check(config, method):
    seed_all(42)
    device = torch.device(config['device'])
    config = dict(config, method=method)
    model = make_model(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    b, s = config['batch_size'], config['crop_size']
    image = torch.randn(b, 3, s, s, device=device)
    mask = torch.randint(4, (b, s, s), device=device)
    d = GaussianDiffusion(config['timesteps'], device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    x0 = d.encode(mask) if method == 'diffusion' else None
    t = torch.randint(config['timesteps'], (b,), device=device)
    logits = model(image, d.q_sample(x0, t, torch.randn_like(x0)), t) if x0 is not None else model(image)
    loss, _ = losses(logits, mask, config, x0)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    result = {'method': method, 'parameter_count': sum(p.numel() for p in model.parameters()),
              'batch_size': b, 'crop_size': s, 'loss': float(loss.detach()), 'grad_norm': float(grad),
              'seconds_including_first_step_startup': time.perf_counter()-start,
              'peak_allocated_mib': torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else None,
              'peak_reserved_mib': torch.cuda.max_memory_reserved(device)/1024**2 if device.type == 'cuda' else None,
              'status': 'passed_one_synthetic_training_step'}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/pilot.json')
    p.add_argument('--output', type=Path, default=ROOT/'docs/resource_smoke.json')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config['cpu_threads'])
    results = []
    for method in ['direct', 'iterative', 'diffusion']:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            result = check(config, method)
        except torch.cuda.OutOfMemoryError as exc:
            result = {'method': method, 'status': 'out_of_memory', 'error': str(exc)}
        results.append(result)
        print(result, flush=True)
    write_json(args.output, {'results': results, 'limitations': 'Synthetic single step, not convergence or full validation throughput'})
    return 0 if all(x['status'].startswith('passed') for x in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())

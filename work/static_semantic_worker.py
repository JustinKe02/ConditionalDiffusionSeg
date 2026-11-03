"""One isolated benchmark, training, or fixed-best validation subprocess."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import json
import time
import traceback
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from conditional_diffusion.data import MaterialDataset
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.metrics_impl import confusion, summarize, boundary_counts, summarize_boundaries
from conditional_diffusion.runtime import write_json
from conditional_diffusion.static_experiment import (configure, build, train_step, loader_for,
    synchronize, predict, train_static, evaluate_static)


def base_config(mode, batch):
    c = json.loads((ROOT/'configs/pilot_recycling.json').read_text())
    c.update(schema_version=2, mode=mode, method={'S0':'direct','S1':'iterative','S2':'diffusion'}[mode],
             experiment_variant='static_semantic_'+mode.lower(), architecture='static_semantic_v1',
             sampling_steps=5, batch_size=batch, window_batch_size=1, precision='fp32',
             static_initialization_seed=42, timestep_grid=[999,749,500,250,0])
    if mode != 'S2':
        for key in ['timestep_sampling','high_noise_start','training_state','recycle_fraction']:
            c.pop(key, None)
    return c


def benchmark_train(c):
    configure(c)
    model = build(c).train()
    diffusion = GaussianDiffusion(1000, 'cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    dataset = MaterialDataset(c['manifest'], 'train', crop=512, seed=42)
    epoch, iterator = 0, None
    def next_batch():
        nonlocal epoch, iterator
        while True:
            if iterator is None:
                epoch += 1; iterator = iter(loader_for(dataset, c, epoch))
            try:
                image, target, _ = next(iterator)
                # An isolated batch16 measurement uses full16 updates; real training retains final8.
                if len(image) != c['batch_size']:
                    continue
                return image.to('cuda'), target.to('cuda')
            except StopIteration:
                iterator = None
    for _ in range(10):
        image, target = next_batch(); train_step(model, diffusion, optimizer, image, target, c)
    synchronize('cuda'); torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    for _ in range(30):
        image, target = next_batch(); train_step(model, diffusion, optimizer, image, target, c)
    synchronize('cuda')
    seconds = time.perf_counter()-begin
    return dict(status='passed', mode=c['mode'], batch_size=c['batch_size'], warmup_updates=10, timed_updates=30,
                timed_samples=30*c['batch_size'], seconds=seconds, seconds_per_update=seconds/30,
                samples_per_second=30*c['batch_size']/seconds,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                total_memory_mib=torch.cuda.get_device_properties(0).total_memory/1024**2,
                includes='real train IO/crops/augmentation; S1 five-step backward or S2 teacher; loss/backward/clip/AdamW',
                purpose='engineering only; weights discarded; no validation model selection',
                full_batch_timing=True, last_partial_batch_note='batch16 timing uses full16; effect training keeps final8')


def benchmark_windows(c, output):
    configure(c)
    model = build(c).eval()
    diffusion = GaussianDiffusion(1000, 'cuda')
    dataset = MaterialDataset(c['manifest'], 'val', augment=False)
    begin = time.perf_counter(); image, target, ident = dataset[0]
    read_seconds = time.perf_counter()-begin
    # Warmup shapes without using a label to predict. One whole image is then timed.
    predict(model, diffusion, image[:, :512, :512], ident, c)
    torch.cuda.reset_peak_memory_stats(); synchronize('cuda'); begin = time.perf_counter()
    prediction, calls, timing, probabilities = predict(model, diffusion, image, ident, c, True)
    synchronize('cuda'); seconds = time.perf_counter()-begin
    begin = time.perf_counter(); matrix = confusion(target.numpy(), prediction)
    bounds = boundary_counts(target.numpy(), prediction, 2)
    metrics_seconds = time.perf_counter()-begin
    np.savez_compressed(output.with_suffix('.npz'), probabilities=probabilities.numpy(), prediction=prediction)
    return dict(status='passed', mode=c['mode'], window_batch_size=c['window_batch_size'],
                image_id=ident, image_shape=list(image.shape), sampling_seed=3407, window_sample_nfe=calls,
                seconds_without_io_or_metrics=seconds, read_seconds=read_seconds, metrics_seconds=metrics_seconds,
                **timing, peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                total_memory_mib=torch.cuda.get_device_properties(0).total_memory/1024**2,
                protocol='fixed initialized weights; val image only for cost/batch-equivalence; metrics not used for model selection')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['benchmark_train','benchmark_windows','train','secondary'])
    p.add_argument('--mode', choices=['S0','S1','S2'], required=True)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    args = p.parse_args()
    c = base_config(args.mode, args.batch)
    try:
        if args.action == 'benchmark_train':
            result = benchmark_train(c)
        elif args.action == 'benchmark_windows':
            c['window_batch_size'] = args.batch
            result = benchmark_windows(c, args.output)
        else:
            plan = json.loads(args.plan.read_text()); c = plan['configs'][args.mode]
            run = Path(plan['runs'][args.mode])
            if args.action == 'train':
                train_static(c, ROOT, run, plan['source_sha256'], resume=run.exists())
                result = dict(status='completed', run=str(run))
            else:
                assert args.mode == 'S2'
                configure(c); model = build(c)
                state = torch.load(run/'checkpoints/best.pt', map_location='cpu', weights_only=False)
                assert state['config'] == c and state['source_sha256'] == plan['source_sha256']
                model.load_state_dict(state['model']); c = dict(c, sampling_seed=3408)
                dataset = MaterialDataset(c['manifest'], 'val', augment=False)
                metrics, matrices, previews, breakdown = evaluate_static(model, GaussianDiffusion(1000, 'cuda'), dataset, c)
                np.savez_compressed(args.output.with_suffix('.npz'), **matrices)
                write_json(args.output.with_name(args.output.stem+'_timing.json'), breakdown)
                result = dict(status='completed', epoch=state['epoch'], sampling_seed=3408, metrics=metrics,
                              purpose='same primary-selected best; no re-selection; not another training seed')
        write_json(args.output, result)
    except BaseException as exc:
        kind = 'oom' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'failed'
        write_json(args.output, dict(status=kind, mode=args.mode, batch=args.batch, error=str(exc), traceback=traceback.format_exc()))
        if kind != 'oom':
            raise
        print('Recorded isolated CUDA OOM', args.mode, args.batch, flush=True)


if __name__ == '__main__':
    main()

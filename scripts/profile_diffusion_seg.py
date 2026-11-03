"""Measure real-data FP32 training throughput in isolated CUDA processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def measure(args):
    import torch
    from torch.utils.data import DataLoader
    from conditional_diffusion.diffusion_seg_experiment import (
        SegmentationDataset, build, collate, configure, train_step,
    )

    config = json.loads(args.config.read_text())
    config.update(batch_size=args.batch_size, workers=args.workers)
    configure(config)
    model, diffusion = build(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    dataset = SegmentationDataset(config, 'train')
    dataset.epoch = 1
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, collate_fn=collate,
                        generator=torch.Generator().manual_seed(config['seed'] + 1))
    batches = iter(loader)
    model.train()
    durations, counts = [], []
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.warmup + args.measure):
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            batch = next(batches)
        except StopIteration:
            dataset.epoch += 1
            batches = iter(loader)
            batch = next(batches)
        terms, gradient = train_step(model, diffusion, optimizer, batch, config)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if index >= args.warmup:
            durations.append(elapsed)
            counts.append(len(batch['image']))
    free, total = torch.cuda.mem_get_info()
    result = dict(status='passed', batch_size=args.batch_size, workers=args.workers,
                  measured_batches=len(durations), measured_samples=sum(counts),
                  seconds=sum(durations), seconds_per_batch=durations,
                  samples_per_second=sum(counts) / sum(durations),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  total_mib=total / 2**20, final_free_mib=free / 2**20,
                  final_loss=terms['loss_total'], gradient_norm=gradient,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  precision='FP32; TF32 disabled; strict deterministic algorithms',
                  scope='ephemeral profiling updates on train split; weights not saved')
    print(json.dumps(result, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[4, 8, 12, 16])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--measure', type=int, default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    if args.workers < 0 or args.warmup < 0 or args.measure < 1:
        parser.error('Worker and warmup counts must be nonnegative, measure must be positive')
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error('batch-size must be positive')
        measure(args)
        return
    if args.output is None or any(batch < 1 for batch in args.batches):
        parser.error('A report output and positive batch candidates are required')
    if args.output.exists():
        parser.error('Profiling report already exists; choose a new output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(config=str(args.config.resolve()),
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(), results=[])
    failed = False
    for batch_size in args.batches:
        command = [sys.executable, '-B', __file__, '--config', str(args.config),
                   '--batch-size', str(batch_size), '--workers', str(args.workers),
                   '--warmup', str(args.warmup), '--measure', str(args.measure)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            error = completed.stderr[-4000:]
            status = 'oom' if 'out of memory' in error.lower() else 'error'
            result = dict(status=status, batch_size=batch_size, workers=args.workers,
                          exit_code=completed.returncode, error=error)
            failed = failed or status == 'error'
        else:
            result = json.loads(completed.stdout.strip().splitlines()[-1])
        report['results'].append(result)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)
        print(json.dumps(result, allow_nan=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

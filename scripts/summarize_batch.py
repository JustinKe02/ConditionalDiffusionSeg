"""Summarize measured validation timing, storage, and completed P1 validation results."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_diffusion.runtime import write_json, directory_size, source_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-id', required=True)
    p.add_argument('--benchmark-only', action='store_true')
    args = p.parse_args()
    if not args.batch_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.batch_id):
        p.error('Invalid batch ID')
    batch = ROOT/'outputs/launches'/args.batch_id
    run = ROOT/'outputs/P0/diffusion/seed_42'/args.batch_id
    evaluations = sorted((run/'evaluations').glob('val_*'))
    if not evaluations:
        raise ValueError('No full validation benchmark found')
    evaluation = evaluations[-1]
    metrics = json.loads((evaluation/'metrics.json').read_text())
    evaluated = json.loads((evaluation/'evaluation_config.json').read_text())
    config = evaluated['config']
    protocol_ok = (config['crop_size'] == 512 and config['stride'] == 384
                   and config['sampling_steps'] == 20 and config['samples_per_image'] == 1
                   and config['sampling_seed'] == 3407)
    gate = json.loads((batch/'p0_gate.json').read_text())
    summary = {'passed': metrics['image_count'] == 25 and metrics['nfe_total'] == 17500 and protocol_ok,
               'evaluation_directory': str(evaluation), 'image_count': metrics['image_count'],
               'seconds': metrics['seconds'], 'seconds_per_image': metrics['seconds']/metrics['image_count'],
               'nfe_total': metrics['nfe_total'], 'peak_allocated_mib': metrics['peak_allocated_mib'],
               'peak_reserved_mib': metrics['peak_reserved_mib'],
               'four_pilot_diffusion_validations_seconds': 4*metrics['seconds'],
               'p0_checkpoint_and_log_measurements': [
                   {k: r[k] for k in ['method', 'sampling_seed', 'checkpoint_bytes', 'last_checkpoint_bytes', 'tensorboard_bytes']}
                   for r in gate['results']],
               'project_allocated_gib': directory_size(ROOT)/1024**3,
               'source_sha256': source_hashes(ROOT),
               'scope': 'full 25-image val timing at 512/384, 20 DDIM steps; P0 weights; scores are not research results',
               'limitations': 'Measured validation cost includes I/O and boundary metrics; does not estimate training duration from synthetic steps.'}
    write_json(batch/'benchmark_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary['passed']:
        raise SystemExit('Benchmark protocol/count mismatch')
    if args.benchmark_only:
        return
    results = []
    for method in ['direct', 'iterative', 'diffusion']:
        pilot = ROOT/'outputs/P1'/method/'seed_42'/args.batch_id
        status = json.loads((pilot/'status.json').read_text())
        if status['status'] != 'completed' or status['last_completed_epoch'] != 40:
            raise ValueError(f'P1 incomplete: {pilot}')
        results.append({'method': method, 'run': str(pilot), 'status': status,
                        'best_val': json.loads((pilot/'best_val_metrics.json').read_text()),
                        'run_size_mib': directory_size(pilot)/1024**2})
    write_json(batch/'pilot_summary.json', {'results': results, 'scope': 'P1 seed42 validation pilot only; no test evaluation'})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()

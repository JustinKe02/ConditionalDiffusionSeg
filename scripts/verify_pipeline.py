"""Small real-data integration check: logging, resume equivalence, evaluation, plotting."""
import csv
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_diffusion.runtime import write_json


def command(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/pipeline_verification.json')
    args = parser.parse_args()
    stamp = str(time.time_ns())
    name_full, name_resume = 'verify_full_'+stamp, 'verify_resume_'+stamp
    base = ROOT/'outputs/debug/diffusion/seed_42'
    continuous, resumed = base/name_full, base/name_resume
    for name, extra in [(name_full, []), (name_resume, ['--stop-after-epoch', '1'])]:
        command(ROOT/'scripts/train.py', '--config', ROOT/'configs/debug.json', '--method', 'diffusion', '--run-id', name, *extra)
    # Simulate log events beyond the last complete checkpoint, then verify reconciliation.
    for category, directory, tag, axis in [('steps', 'steps', 'train_step/loss_total', 3), ('val', 'epochs', 'val/miou_4class', 3)]:
        path = resumed/f'metrics_{category}.csv'
        with path.open('a', newline='') as f:
            csv.writer(f).writerow([time.time(), 3, 3, tag, 999.])
        writer = SummaryWriter(str(resumed/'tensorboard'/directory))
        writer.add_scalar(tag, 999., axis)
        writer.close()
    command(ROOT/'scripts/train.py', '--resume', resumed/'checkpoints/last.pt')
    a = torch.load(continuous/'checkpoints/last.pt', map_location='cpu', weights_only=False)
    b = torch.load(resumed/'checkpoints/last.pt', map_location='cpu', weights_only=False)
    assert a['epoch'] == b['epoch'] == 2 and a['global_step'] == b['global_step'] == 2
    assert all(torch.equal(v, b['model'][k]) for k, v in a['model'].items()), 'Resume weights differ'
    for parameter, states in a['optimizer']['state'].items():
        for key, value in states.items():
            assert torch.equal(value, b['optimizer']['state'][parameter][key])
    assert a['scheduler'] == b['scheduler']
    for category, directory, tag in [('steps', 'steps', 'train_step/loss_total'), ('val', 'epochs', 'val/miou_4class')]:
        with (resumed/f'metrics_{category}.csv').open() as f:
            rows = [r for r in csv.DictReader(f) if r['tag'] == tag]
        assert [int(r['epoch']) for r in rows] == [1, 2]
        events = EventAccumulator(str(resumed/'tensorboard'/directory)).Reload().Scalars(tag)
        assert [e.step for e in events] == [1, 2]
        assert all(abs(e.value-float(r['value'])) < 1e-6 for e, r in zip(events, rows))
    command(ROOT/'scripts/evaluate.py', '--checkpoint', continuous/'checkpoints/best.pt', '--split', 'val')
    figures = ROOT/'outputs/debug'/('figures_'+stamp)
    command(ROOT/'scripts/plot_curves.py', '--runs', continuous, resumed, '--output', figures)
    report = {'status': 'passed', 'continuous_run': str(continuous), 'resumed_run': str(resumed),
              'model_weights_bitwise_equal': True, 'optimizer_state_bitwise_equal': True,
              'scheduler_state_equal': True, 'tensorboard_csv_values_match': True,
              'stale_log_tail_reconciled': True, 'evaluation_cli': 'passed_debug_val_only',
              'figures': str(figures), 'scope': '2 training batches at 32x32; no research performance claims'}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

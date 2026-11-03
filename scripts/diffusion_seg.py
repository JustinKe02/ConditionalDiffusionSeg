"""Run one explicit condition-mask diffusion experiment or an engineering smoke."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from conditional_diffusion.diffusion_seg_experiment import (
    evaluate_checkpoint, json_finite, smoke, train, validate_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'evaluate', 'smoke'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--run', type=Path, help='Named directory under this project outputs')
    parser.add_argument('--resume', action='store_true', help='Resume run/last.pt at its committed epoch')
    parser.add_argument('--stop-after-epoch', type=int)
    parser.add_argument('--device', help='Override configured device, useful for CPU smoke checks')
    args = parser.parse_args()
    if args.action != 'train' and (args.resume or args.stop_after_epoch is not None):
        parser.error('--resume and --stop-after-epoch are train options')
    if args.action != 'smoke' and args.run is None:
        parser.error('--run is required for train and evaluate')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.device:
        config['device'] = args.device
    validate_config(config)
    if args.action == 'train':
        result = train(config, args.run, ROOT, args.resume, args.stop_after_epoch)
    elif args.action == 'evaluate':
        result = evaluate_checkpoint(config, args.run, ROOT)
    else:
        result = smoke(config)
    print(json.dumps(json_finite(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

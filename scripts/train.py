"""Train exactly one explicitly selected experiment. No automatic experiment queue."""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_diffusion.trainer import train, validate_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=Path)
    group.add_argument('--resume', type=Path, help='Trusted local full checkpoint only')
    p.add_argument('--method', choices=['direct', 'iterative', 'diffusion'])
    p.add_argument('--seed', type=int)
    p.add_argument('--class-balance', action='store_true')
    p.add_argument('--boundary', action='store_true')
    p.add_argument('--run-id')
    p.add_argument('--stop-after-epoch', type=int, help='Controlled early stop, preserving original total schedule')
    args = p.parse_args()
    if args.resume:
        if args.method or args.seed is not None or args.class_balance or args.boundary or args.run_id:
            p.error('Resume cannot change the frozen configuration or run ID')
        resume = args.resume.resolve()
        run = resume.parent.parent
        if not run.is_relative_to(ROOT/'outputs') or resume.name != 'last.pt':
            p.error('Resume must reference this project outputs/.../checkpoints/last.pt')
        config = json.loads((run/'config.json').read_text())
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['epoch'] >= config['epochs']:
            p.error('This run is already complete')
    else:
        resume = None
        config = json.loads(args.config.read_text())
        if args.method:
            config['method'] = args.method
        if args.seed is not None:
            config['seed'] = args.seed
        config['class_balance'] = config['class_balance'] or args.class_balance
        config['boundary'] = config['boundary'] or args.boundary
        validate_config(config)
        ident = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', ident):
            p.error('Invalid run ID')
        method = config['method']+('_class' if config['class_balance'] else '')+('_boundary' if config['boundary'] else '')
        phase = 'P0' if config.get('fixed_crops') else ('debug' if config['debug'] else config['phase'])
        if not re.fullmatch(r'[A-Za-z0-9_-]+', phase):
            p.error('Invalid phase')
        run = ROOT/'outputs'/phase/method/f"seed_{config['seed']}"/ident
        run.mkdir(parents=True, exist_ok=False)
    train(config, ROOT, run, resume, args.stop_after_epoch)


if __name__ == '__main__':
    main()

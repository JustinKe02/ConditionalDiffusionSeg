"""Require memorization from independent initial noise before proceeding to P1."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_diffusion.data import FixedCropDataset, CLASSES
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.inference import evaluate
from conditional_diffusion.model import make_model
from conditional_diffusion.runtime import StorageGuard, seed_all, write_json, directory_size, source_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs=3, type=Path, required=True, help='direct iterative diffusion runs in this order')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    thresholds = {'miou_4class': .90, 'minimum_class_iou': .75}
    results = []
    for method, run in zip(['direct', 'iterative', 'diffusion'], args.runs):
        status = json.loads((run/'status.json').read_text())
        if status['status'] != 'completed':
            raise ValueError(f'Incomplete P0 run: {run}')
        metadata = json.loads((run/'run_metadata.json').read_text())
        if metadata['source_sha256'] != source_hashes(ROOT):
            raise ValueError(f'Code changed since P0 training: {run}')
        checkpoint = run/'checkpoints/best.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        c = state['config']
        assert c['method'] == method and c['debug'] and c['phase'] == 'P0' and c['fixed_crops']
        torch.set_num_threads(c['cpu_threads'])
        seed_all(c['seed'])
        guard = StorageGuard(ROOT, run, c['storage'])
        guard.check(16*1024**2)
        ds = FixedCropDataset(c['manifest'], c['fixed_crops'])
        model = make_model(c).to(c['device'])
        model.load_state_dict(state['model'])
        diffusion = GaussianDiffusion(c['timesteps'], c['device'])
        seeds = [3407, 3408] if method == 'diffusion' else [3407]
        for seed in seeds:
            config = dict(c, sampling_seed=seed)
            metrics, matrices, previews = evaluate(model, diffusion, ds, config)
            minimum = min(metrics['iou_'+cls] for cls in CLASSES)
            passed = metrics['miou_4class'] >= thresholds['miou_4class'] and minimum >= thresholds['minimum_class_iou']
            detail = {'method': method, 'sampling_seed': seed, 'checkpoint': str(checkpoint.resolve()),
                      'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      'checkpoint_bytes': checkpoint.stat().st_size,
                      'last_checkpoint_bytes': (run/'checkpoints/last.pt').stat().st_size,
                      'tensorboard_bytes': directory_size(run/'tensorboard'),
                      'metrics': metrics, 'passed': passed}
            results.append(detail)
            out = run/f'p0_sampling_seed_{seed}'
            out.mkdir(exist_ok=True)
            from PIL import Image
            for ident, panels in previews.items():
                Image.fromarray(np.concatenate(list(panels.values()), axis=1)).save(out/(ident+'.png'))
            np.savez_compressed(out/'confusion.npz', **matrices)
            write_json(out/'metrics.json', detail)
            print(json.dumps(detail), flush=True)
        del model
        torch.cuda.empty_cache()
    report = {'passed': all(r['passed'] for r in results), 'thresholds': thresholds, 'results': results,
              'source_sha256': source_hashes(ROOT),
              'fixed_crops_sha256': hashlib.sha256(Path(c['fixed_crops']).read_bytes()).hexdigest(),
              'scope': 'fixed TRAIN crop memorization; diffusion predictions start from independent Gaussian noise; not thesis results'}
    write_json(args.output, report)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

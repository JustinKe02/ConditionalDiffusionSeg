"""Evaluate a trusted local checkpoint. Test requires a frozen sampling protocol."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_diffusion.data import MaterialDataset
from conditional_diffusion.reference_data import dataset_from_config
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.model import make_model
from conditional_diffusion.inference import evaluate
from conditional_diffusion.runtime import StorageGuard, write_json, seed_all, source_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--steps', type=int)
    p.add_argument('--samples', type=int)
    p.add_argument('--sampling-seed', type=int)
    p.add_argument('--protocol', type=Path, help='Frozen JSON: sampling_steps, samples_per_image, sampling_seed')
    p.add_argument('--allow-test', action='store_true')
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--full-val', action='store_true', help='Benchmark all original val images even for a P0/debug checkpoint')
    p.add_argument('--window-size', type=int, help='Window override for --full-val benchmarking only')
    p.add_argument('--stride', type=int, help='Stride override for --full-val benchmarking only')
    args = p.parse_args()
    if args.full_val and args.split != 'val':
        p.error('--full-val is restricted to the validation split')
    if (args.window_size is not None or args.stride is not None) and not args.full_val:
        p.error('Spatial overrides require --full-val')
    checkpoint = args.checkpoint.resolve()
    run = checkpoint.parent.parent
    if not run.is_relative_to(ROOT/'outputs'):
        p.error('Checkpoint must belong to this project outputs')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = dict(state['config'])
    if args.window_size is not None:
        config['crop_size'] = args.window_size
    if args.stride is not None:
        config['stride'] = args.stride
    if config['crop_size'] < 16 or config['crop_size'] % 8 or not 0 < config['stride'] <= config['crop_size']:
        p.error('Invalid window/stride')
    if args.split == 'test' and (not args.allow_test or not args.protocol or config['debug']):
        p.error('Test requires --allow-test, --protocol, and a non-debug checkpoint')
    if args.protocol:
        if any(x is not None for x in [args.steps, args.samples, args.sampling_seed]):
            p.error('Do not override a frozen protocol')
        protocol = json.loads(args.protocol.read_text())
        if set(protocol) != {'sampling_steps', 'samples_per_image', 'sampling_seed'}:
            p.error('Protocol must contain exactly the three documented sampling fields')
        config.update(protocol)
    else:
        for key, value in [('sampling_steps', args.steps), ('samples_per_image', args.samples), ('sampling_seed', args.sampling_seed)]:
            if value is not None:
                config[key] = value
    if not (1 <= config['sampling_steps'] <= config['timesteps']) or config['samples_per_image'] < 1 or config['sampling_seed'] < 0:
        p.error('Invalid sampling configuration')
    torch.set_num_threads(config['cpu_threads'])
    seed_all(config['seed'])
    guard = StorageGuard(ROOT, run, config['storage'])
    guard.check(16*1024**2)
    out = run/'evaluations'/f'{args.split}_{time.time_ns()}'
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(config['device'])
    model = make_model(config).to(device)
    model.load_state_dict(state['model'])
    subset = config['debug'] and not args.full_val
    ds = dataset_from_config(config, args.split, crop=config['crop_size'] if subset else None,
                         seed=config['seed'], augment=False, limit=config['val_limit'] if subset else None)
    def save(ident, prediction):
        guard.check(prediction.nbytes+1024**2)
        if args.full_val:
            print(f'[full-val] completed {ident}', flush=True)
        if args.save_predictions:
            Image.fromarray(prediction).save(out/(ident+'.png'))
    result, matrices, previews = evaluate(model, GaussianDiffusion(config['timesteps'], device), ds, config, on_image=save)
    if args.full_val:
        for ident, panels in previews.items():
            thumbnails = []
            for name, arr in panels.items():
                im = Image.fromarray(arr)
                im.thumbnail((512, 512), Image.Resampling.BILINEAR if name == 'rgb' else Image.Resampling.NEAREST)
                thumbnails.append(np.asarray(im))
            guard.check(1024**2)
            Image.fromarray(np.concatenate(thumbnails, axis=1)).save(out/(ident+'_preview.png'))
    guard.check(1024**2)
    np.savez_compressed(out/'confusion_matrices.npz', **matrices)
    write_json(out/'metrics.json', result)
    write_json(out/'evaluation_config.json', {'checkpoint': str(checkpoint), 'checkpoint_epoch': state['epoch'],
               'split': args.split, 'config': config, 'source_sha256': source_hashes(ROOT),
               'scope': 'full_val_timing_only' if args.full_val and config['debug'] else 'evaluation',
               'protocol_path': None if args.protocol is None else str(args.protocol.resolve())})
    print(json.dumps({'output': str(out), 'metrics': result}, indent=2))


if __name__ == '__main__':
    main()

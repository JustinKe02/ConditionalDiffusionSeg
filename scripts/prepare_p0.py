"""Freeze four boundary-containing, class-covering crops using training labels only."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_diffusion.data import FixedCropDataset, read_manifest
from conditional_diffusion.inference import positions
from conditional_diffusion.runtime import write_json


def choose_crops(manifest_path, size):
    manifest = read_manifest(manifest_path)
    ids = Path(manifest['splits']['train']['snapshot_path']).read_text().split()
    candidates = []
    for ident in ids:
        with Image.open(Path(manifest['dataset_root'])/manifest['mask_pattern'].format(id=ident)) as im:
            mask = np.asarray(im)
        ys = np.array(positions(mask.shape[0], size, size//2))
        xs = np.array(positions(mask.shape[1], size, size//2))
        counts = []
        for cls in range(4):
            integral = np.pad((mask == cls).astype(np.int32), ((1, 0), (1, 0)))
            integral = integral.cumsum(0).cumsum(1)
            counts.append(integral[np.ix_(ys+size, xs+size)]-integral[np.ix_(ys, xs+size)]
                          -integral[np.ix_(ys+size, xs)]+integral[np.ix_(ys, xs)])
        counts = np.stack(counts, -1)
        fractions = counts/(size*size)
        for cls in range(4):
            # The target class and its complement each occupy substantial crop area.
            score = fractions[..., cls]*(1-fractions[..., cls])
            row, col = np.unravel_index(score.argmax(), score.shape)
            candidates.append({'id': ident, 'top': int(ys[row]), 'left': int(xs[col]),
                               'pixel_counts': counts[row, col].tolist(),
                               'target_class': cls, 'selection_score': float(score[row, col])})
    chosen, used = [], set()
    for cls in [1, 2, 3, 0]:
        ranked = sorted((c for c in candidates if c['target_class'] == cls and c['id'] not in used),
                        key=lambda c: (-c['selection_score'], c['id'], c['top'], c['left']))
        chosen.append(ranked[0])
        used.add(ranked[0]['id'])
    counts = np.sum([c['pixel_counts'] for c in chosen], axis=0)
    if np.any(counts < size*size*.1):
        raise ValueError('Insufficient class coverage for P0')
    return {'schema_version': 1, 'split': 'train', 'crop_size': size,
            'train_split_sha256': manifest['splits']['train']['sha256'], 'augmentation': 'none',
            'selection': 'deterministic label-only grid; one distinct image per target class; maximize p*(1-p)',
            'crops': chosen, 'total_pixel_counts': counts.tolist(),
            'scope': 'memorization diagnostic only; not validation or thesis performance'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'data_manifests/source_dataset.json')
    p.add_argument('--output', type=Path, default=ROOT/'data_manifests/p0_fixed_crops.json')
    p.add_argument('--crop-size', type=int, default=256)
    args = p.parse_args()
    if args.crop_size < 16 or args.crop_size % 8:
        p.error('Crop size must be >=16 and divisible by 8')
    if not args.output.exists():
        write_json(args.output, choose_crops(args.manifest, args.crop_size))
    ds = FixedCropDataset(args.manifest, args.output)
    if ds.crop != args.crop_size:
        p.error('Existing fixed crop size differs; do not overwrite a frozen specification')
    print(args.output.read_text(), flush=True)


if __name__ == '__main__':
    main()

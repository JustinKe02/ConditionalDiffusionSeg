import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]
CLASSES = ['background', 'monolayer', 'fewlayer', 'multilayer']
PALETTE = np.array([[0, 0, 0], [0, 170, 255], [255, 190, 0], [220, 50, 90]], dtype=np.uint8)


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    all_ids = []
    for split in manifest['splits'].values():
        payload = Path(split['snapshot_path']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != split['sha256']:
            raise ValueError('Split snapshot hash mismatch')
        ids = payload.decode().split()
        if len(ids) != split['count'] or len(set(ids)) != len(ids):
            raise ValueError('Invalid split count or duplicate ID')
        all_ids.extend(ids)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Cross-split duplicate IDs')
    return manifest


class MaterialDataset(Dataset):
    def __init__(self, manifest, split, crop=None, seed=42, augment=True, limit=None):
        self.manifest = read_manifest(manifest)
        self.root = Path(self.manifest['dataset_root'])
        self.ids = Path(self.manifest['splits'][split]['snapshot_path']).read_text().split()
        if limit is not None:
            self.ids = self.ids[:limit]
        self.crop, self.seed, self.epoch, self.augment = crop, seed, 0, augment

    def __len__(self):
        return len(self.ids)

    def raw(self, index):
        ident = self.ids[index]
        with Image.open(self.root / self.manifest['image_pattern'].format(id=ident)) as im:
            rgb = np.asarray(im.convert('RGB')).copy()
        with Image.open(self.root / self.manifest['mask_pattern'].format(id=ident)) as im:
            mask = np.asarray(im).copy()
        if mask.ndim != 2 or rgb.shape[:2] != mask.shape or mask.min() < 0 or mask.max() > 3:
            raise ValueError(f'Invalid image/mask pair: {ident}')
        return rgb, mask, ident

    def __getitem__(self, index):
        rgb, mask, ident = self.raw(index)
        if self.crop:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            h, w = mask.shape
            pad = ((0, max(0, self.crop-h)), (0, max(0, self.crop-w)))
            rgb = np.pad(rgb, (*pad, (0, 0)), mode='edge')
            mask = np.pad(mask, pad, mode='edge')
            y = int(rng.integers(rgb.shape[0]-self.crop+1))
            x = int(rng.integers(rgb.shape[1]-self.crop+1))
            rgb, mask = rgb[y:y+self.crop, x:x+self.crop], mask[y:y+self.crop, x:x+self.crop]
            if self.augment:
                k = int(rng.integers(4))
                rgb, mask = np.rot90(rgb, k), np.rot90(mask, k)
                for axis in [0, 1]:
                    if rng.random() < .5:
                        rgb, mask = np.flip(rgb, axis), np.flip(mask, axis)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()/255
        return (image-MEAN)/STD, torch.from_numpy(np.ascontiguousarray(mask)).long(), ident


class FixedCropDataset(Dataset):
    """Train-only, cached crops for the P0 memorization diagnostic. No augmentation."""
    def __init__(self, manifest, specification):
        self.manifest = read_manifest(manifest)
        spec = json.loads(Path(specification).read_text())
        if spec['train_split_sha256'] != self.manifest['splits']['train']['sha256']:
            raise ValueError('Fixed crops belong to a different training split')
        self.crop = spec['crop_size']
        source = MaterialDataset(manifest, 'train', augment=False)
        self.items, self.ids = [], []
        for item in spec['crops']:
            if item['id'] not in source.ids:
                raise ValueError('P0 crops must come exclusively from train')
            rgb, mask, _ = source.raw(source.ids.index(item['id']))
            y, x = item['top'], item['left']
            if min(y, x) < 0 or y+self.crop > mask.shape[0] or x+self.crop > mask.shape[1]:
                raise ValueError('Fixed crop outside image')
            rgb = np.ascontiguousarray(rgb[y:y+self.crop, x:x+self.crop])
            mask = np.ascontiguousarray(mask[y:y+self.crop, x:x+self.crop])
            if np.bincount(mask.ravel(), minlength=4).tolist() != item['pixel_counts']:
                raise ValueError('Fixed crop labels changed')
            ident = f"{item['id']}_y{y}_x{x}"
            image = torch.from_numpy(rgb).permute(2, 0, 1).float()/255
            self.items.append(((image-MEAN)/STD, torch.from_numpy(mask).long(), ident))
            self.ids.append(ident)
        if len(self.items) != 4 or len(set(self.ids)) != 4:
            raise ValueError('P0 requires four distinct fixed crops')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def audit_dataset(manifest):
    result = {'splits': {}, 'provenance_independence': 'not_verified_by_file_audit'}
    for split in ['train', 'val', 'test']:
        ds = MaterialDataset(manifest, split)
        counts = np.zeros(4, dtype=np.int64)
        sizes = set()
        for i in range(len(ds)):
            rgb, mask, _ = ds.raw(i)
            counts += np.bincount(mask.ravel(), minlength=4)
            sizes.add((rgb.shape[1], rgb.shape[0]))
        result['splits'][split] = {'count': len(ds), 'pixel_counts': counts.tolist(), 'sizes_wh': sorted(sizes)}
    return result

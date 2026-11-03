"""RGB-only cached reference fields, evaluated before synchronized crop augmentation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import MaterialDataset, MEAN, STD


def reference_patch(rgb, entry, top=0, left=0, *, contrast_weighted=True):
    """Evaluate a full-image RGB fit at patch coordinates; never accepts a label."""
    h, w = rgb.shape[:2]
    full_w, full_h = entry['size_wh']
    yy = (2 * np.minimum(np.arange(top, top + h), full_h - 1) / (full_h - 1) - 1).astype(np.float32)[:, None]
    xx = (2 * np.minimum(np.arange(left, left + w), full_w - 1) / (full_w - 1) - 1).astype(np.float32)[None, :]
    beta = np.asarray(entry['coefficient_rgb'], dtype=np.float32)
    reference = np.empty((h, w, 3), dtype=np.float32)
    for c in range(3):
        reference[..., c] = (beta[0, c] + beta[1, c] * xx + beta[2, c] * yy
                             + beta[3, c] * xx * xx + beta[4, c] * xx * yy + beta[5, c] * yy * yy)
    np.clip(reference, 0, 1, out=reference)
    q = float(entry['rgb_confidence']) if entry['accepted'] else 0.0
    contrast = np.clip((rgb.astype(np.float32) / 255 - reference) / np.maximum(reference, .02), -1, 1)
    auxiliary = np.empty((h, w, 4), dtype=np.float32)
    auxiliary[..., :3] = contrast * (q if contrast_weighted else float(q > 0))
    auxiliary[..., 3] = q
    return auxiliary


class ReferenceMaterialDataset(MaterialDataset):
    def __init__(self, *args, reference_cache, reference_cache_sha256, reference_enabled,
                 reference_contrast_weighted=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(reference_enabled, bool):
            raise ValueError('reference_enabled must be boolean')
        payload = Path(reference_cache).read_bytes()
        if hashlib.sha256(payload).hexdigest() != reference_cache_sha256:
            raise ValueError('Frozen reference cache hash mismatch')
        cache = json.loads(payload)
        if cache['schema'] != 'rgb_reference_cache_v1':
            raise ValueError('Unknown reference cache schema')
        self.references = cache['images']
        self.reference_enabled = reference_enabled
        self.reference_contrast_weighted = reference_contrast_weighted
        if not set(self.ids).issubset(self.references):
            raise ValueError('Reference cache does not cover this split; no automatic estimation or test exposure')

    def __getitem__(self, index):
        rgb, mask, ident = self.raw(index)
        entry = self.references[ident]
        if [rgb.shape[1], rgb.shape[0]] != entry['size_wh']:
            raise ValueError('Reference image dimensions changed')
        if self.crop:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            h, w = mask.shape
            pad = ((0, max(0, self.crop - h)), (0, max(0, self.crop - w)))
            rgb = np.pad(rgb, (*pad, (0, 0)), mode='edge')
            mask = np.pad(mask, pad, mode='edge')
            y = int(rng.integers(rgb.shape[0] - self.crop + 1))
            x = int(rng.integers(rgb.shape[1] - self.crop + 1))
            rgb, mask = rgb[y:y + self.crop, x:x + self.crop], mask[y:y + self.crop, x:x + self.crop]
            aux = reference_patch(rgb, entry, y, x, contrast_weighted=self.reference_contrast_weighted) if self.reference_enabled else np.zeros((*mask.shape, 4), np.float32)
            if self.augment:
                k = int(rng.integers(4))
                rgb, mask, aux = np.rot90(rgb, k), np.rot90(mask, k), np.rot90(aux, k)
                for axis in [0, 1]:
                    if rng.random() < .5:
                        rgb, mask, aux = np.flip(rgb, axis), np.flip(mask, axis), np.flip(aux, axis)
        else:
            aux = reference_patch(rgb, entry, contrast_weighted=self.reference_contrast_weighted) if self.reference_enabled else np.zeros((*mask.shape, 4), np.float32)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255
        extra = torch.from_numpy(np.ascontiguousarray(aux)).permute(2, 0, 1)
        return torch.cat([(image - MEAN) / STD, extra]), torch.from_numpy(np.ascontiguousarray(mask)).long(), ident


def dataset_from_config(config, split, **kwargs):
    if config.get('condition_architecture') in ['split_rgb_state', 'frozen_reference_adapter']:
        return ReferenceMaterialDataset(config['manifest'], split, reference_cache=config['reference_cache'],
                    reference_cache_sha256=config['reference_cache_sha256'], reference_enabled=config['reference_enabled'],
                    reference_contrast_weighted=config.get('reference_contrast_weighted', True), **kwargs)
    return MaterialDataset(config['manifest'], split, **kwargs)

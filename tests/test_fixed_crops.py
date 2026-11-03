import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_diffusion.data import FixedCropDataset
from conditional_diffusion.trainer import validate_config


class FixedCropTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mask = (np.indices((48, 64))[1] % 4).astype(np.uint8)
        splits = {}
        for split, ids in [('train', ['a', 'b', 'c', 'd']), ('val', ['v']), ('test', ['t'])]:
            path = self.root/(split+'.txt')
            path.write_text('\n'.join(ids)+'\n')
            splits[split] = {'snapshot_path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'count': len(ids)}
            for ident in ids:
                Image.fromarray(np.full((48, 64, 3), 100, dtype=np.uint8)).save(self.root/(ident+'.jpg'))
                Image.fromarray(self.mask).save(self.root/(ident+'.png'))
        manifest = {'dataset_root': str(self.root), 'splits': splits,
                    'image_pattern': '{id}.jpg', 'mask_pattern': '{id}.png'}
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps(manifest))
        self.spec = {'train_split_sha256': splits['train']['sha256'], 'crop_size': 32,
                     'crops': [{'id': ident, 'top': 16, 'left': 0, 'pixel_counts': [256]*4} for ident in ['a', 'b', 'c', 'd']]}
        self.path = self.root/'crops.json'

    def dataset(self):
        self.path.write_text(json.dumps(self.spec))
        return FixedCropDataset(self.manifest, self.path)

    def test_crops_stay_fixed_across_epochs(self):
        ds = self.dataset()
        before = ds[0]
        ds.epoch = 99
        after = ds[0]
        self.assertEqual(before[2], 'a_y16_x0')
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(np.array_equal(before[1].numpy(), self.mask[16:48, :32]))

    def test_reject_holdout_and_changed_labels(self):
        self.spec['crops'][0]['id'] = 'v'
        with self.assertRaisesRegex(ValueError, 'exclusively from train'):
            self.dataset()
        self.spec['crops'][0]['id'] = 'a'
        self.spec['crops'][0]['pixel_counts'][0] = 0
        with self.assertRaisesRegex(ValueError, 'labels changed'):
            self.dataset()

    def test_formal_config_cannot_use_fixed_train_evaluation(self):
        config = json.loads((ROOT/'configs/pilot.json').read_text())
        config['fixed_crops'] = str(self.path)
        with self.assertRaisesRegex(ValueError, 'restricted to the P0'):
            validate_config(config)


if __name__ == '__main__':
    unittest.main()

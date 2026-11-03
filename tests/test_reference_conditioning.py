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
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.data import MaterialDataset
from conditional_diffusion.reference_data import ReferenceMaterialDataset, reference_patch
from conditional_diffusion.model import make_model
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.objectives import losses
from conditional_diffusion.trainer import validate_config


class ReferenceConditioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        yy, xx = np.mgrid[:40, :48]
        self.rgb = np.stack([80 + xx, 90 + yy, 100 + (xx + yy) // 2], axis=-1).astype(np.uint8)
        target = ((xx // 12 + yy // 10) % 4).astype(np.uint8)
        splits, refs = {}, {}
        for name, ids in [('train', ['a', 'b']), ('val', ['c']), ('test', ['d'])]:
            path = self.root / (name + '.txt')
            path.write_text('\n'.join(ids) + '\n')
            splits[name] = dict(snapshot_path=str(path), count=len(ids), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for ident in ids:
                Image.fromarray(self.rgb).save(self.root / (ident + '.png'))
                Image.fromarray(target).save(self.root / (ident + '_mask.png'))
                if name != 'test':
                    refs[ident] = dict(size_wh=[48, 40], accepted=ident != 'b', rgb_confidence=.8,
                        coefficient_rgb=[[.5, .5, .5], [.01, .02, .03], [.02, .01, -.01], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps(dict(splits=splits, dataset_root=str(self.root),
                                                image_pattern='{id}.png', mask_pattern='{id}_mask.png')))
        self.cache = self.root / 'cache.json'
        self.cache.write_text(json.dumps(dict(schema='rgb_reference_cache_v1', images=refs)))
        self.digest = hashlib.sha256(self.cache.read_bytes()).hexdigest()
        self.entry = refs['a']

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, enabled=True, **kwargs):
        return ReferenceMaterialDataset(self.manifest, 'train', reference_cache=self.cache,
                    reference_cache_sha256=self.digest, reference_enabled=enabled, **kwargs)

    def config(self, enabled=False):
        c = json.loads((ROOT / 'configs/pilot_reference_f0.json').read_text())
        c.update(reference_enabled=enabled, experiment_variant='reference_fusion_f1' if enabled else 'reference_fusion_f0')
        return c

    def test_crop_uses_same_global_field_and_synchronized_geometry(self):
        full = reference_patch(self.rgb, self.entry)
        self.assertTrue(np.allclose(reference_patch(self.rgb[5:37, 9:41], self.entry, 5, 9), full[5:37, 9:41], atol=1e-7))
        ds = self.dataset(crop=32, seed=42); ds.epoch = 7
        old = MaterialDataset(self.manifest, 'train', crop=32, seed=42); old.epoch = 7
        image, target, ident = ds[0]; previous, old_target, old_ident = old[0]
        self.assertTrue(torch.equal(image[:3], previous)); self.assertTrue(torch.equal(target, old_target))
        self.assertEqual(ident, old_ident)
        rng = np.random.default_rng(np.random.SeedSequence([42, 7, 0]))
        y, x = int(rng.integers(9)), int(rng.integers(17))
        expected = full[y:y + 32, x:x + 32]
        expected = np.rot90(expected, int(rng.integers(4)))
        for axis in [0, 1]:
            if rng.random() < .5:
                expected = np.flip(expected, axis)
        self.assertTrue(np.allclose(image[3:].permute(1, 2, 0).numpy(), expected, atol=1e-7))

    def test_reference_features_do_not_depend_on_target_labels(self):
        ds = self.dataset(crop=32, seed=42)
        image, target, _ = ds[0]
        Image.fromarray(np.full((40, 48), 3, np.uint8)).save(self.root / 'a_mask.png')
        new_image, new_target, _ = ds[0]
        self.assertTrue(torch.equal(image, new_image)); self.assertFalse(torch.equal(target, new_target))

    def test_rejected_references_and_f0_disable_all_auxiliary_channels(self):
        for enabled, index in [(False, 0), (True, 1)]:
            image, _, _ = self.dataset(enabled=enabled, crop=32)[index]
            self.assertEqual(tuple(image.shape), (7, 32, 32))
            self.assertEqual(int(torch.count_nonzero(image[3:])), 0)

    def test_cache_hash_and_test_coverage_enforced(self):
        with self.assertRaises(ValueError):
            ReferenceMaterialDataset(self.manifest, 'test', reference_cache=self.cache,
                    reference_cache_sha256=self.digest, reference_enabled=True)
        self.cache.write_text(self.cache.read_text() + ' ')
        with self.assertRaises(ValueError):
            self.dataset()

    def test_registered_f0_f1_and_reject_mismatched_settings(self):
        validate_config(self.config(False)); validate_config(self.config(True))
        for patch in [dict(reference_enabled=True), dict(condition_fusion='adaptive'), dict(phase='P2'),
                      dict(debug=True), dict(method='direct'), dict(dual_head=True), dict(training_state='ground_truth'),
                      dict(experiment_variant='high_noise_recycling'), dict(reference_cache_sha256='')]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_config(dict(self.config(), **patch))

    def test_f0_f1_identical_initialization_and_rng_and_feature_gradients(self):
        c0, c1 = self.config(False), self.config(True)
        c0['base_channels'] = c1['base_channels'] = 8
        torch.manual_seed(42); a = make_model(c0); rng = torch.get_rng_state()
        torch.manual_seed(42); b = make_model(c1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
        image, xt, t = torch.randn(2, 7, 32, 32), torch.randn(2, 4, 32, 32), torch.tensor([100, 950])
        mask = torch.randint(4, (2, 32, 32))
        out = b(image, xt, t)
        loss, _ = losses(out, mask, c1, GaussianDiffusion.encode(mask)); loss.backward()
        self.assertEqual(tuple(out.shape), (2, 4, 32, 32))
        for param in [b.state_input.weight, b.image_input.weight, b.out[-1].weight]:
            self.assertTrue(torch.isfinite(param.grad).all()); self.assertGreater(float(param.grad.abs().sum()), 0)
        self.assertGreater(float(b.image_input.weight.grad[:, 3:].abs().sum()), 0)

    def test_adaptive_model_ddim_and_recycling_support_seven_channel_image(self):
        from conditional_diffusion.recycling import mixed_recycling_input
        c = self.config(True); c.update(base_channels=8, condition_fusion='adaptive', experiment_variant='reference_fusion_f3')
        validate_config(c); model = make_model(c)
        image = torch.randn(2, 7, 32, 32); image[:, 6] = .7
        d = GaussianDiffusion(1000); x0 = d.encode(torch.randint(4, (2, 32, 32)))
        noisy, selected = mixed_recycling_input(model, d, image, x0, torch.tensor([100, 950]), torch.randn_like(x0))
        self.assertFalse(noisy.requires_grad); self.assertEqual(selected.numel(), 1)
        sampled = d.sample(model, image, torch.randn_like(x0), 3)
        self.assertEqual(tuple(sampled.shape), (2, 4, 32, 32)); self.assertTrue(torch.isfinite(sampled).all())


if __name__ == '__main__':
    unittest.main()

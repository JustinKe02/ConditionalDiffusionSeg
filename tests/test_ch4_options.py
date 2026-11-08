"""Unit tests for Chapter 4 G-arm options (vignette, flip_mode, PowerEMA, eval overrides)."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'work'))

from conditional_diffusion import diffusion_seg_experiment as experiment
from conditional_diffusion.data import MaterialDataset
from conditional_diffusion.mask_diffusion import GaussianMaskDiffusion, MaskStateCodec
import glpflow_core as G


class TinyDenoiser(torch.nn.Module):
    def __init__(self, state_channels=3, timesteps=8, **kwargs):
        super().__init__()
        self.head = torch.nn.Conv2d(3, 4, 1)
        self.prior = torch.nn.Conv2d(3, 4, 1)
        self.state_scale = torch.nn.Parameter(torch.tensor(0.1))
        self.timesteps = timesteps
        self.calls = []

    def encode_global(self, thumbnails):
        return [(t.flatten(1).T, self.prior(F.avg_pool2d(t[None], 4))[0], (1, 1)) for t in thumbnails]

    def encode_condition(self, image, boxes, contexts):
        return image, torch.stack([c[1].mean((1, 2)) for c in contexts])

    def denoise(self, condition, state, times):
        self.calls.append((state.detach().cpu().clone(), times.detach().cpu().clone()))
        image, prior = condition
        return self.head(image) + prior[:, :, None, None] + self.state_scale * state.mean(1, keepdim=True)


def make_fixture(project: Path, height=79, width=93, with_test=False):
    data = project / 'data'
    data.mkdir()
    yy, xx = np.indices((height, width))
    mask = ((yy >= height // 2) * 2 + (xx >= width // 2)).astype(np.uint8)
    rgb = np.stack(((mask * 60 + 20) % 255, (xx * 2 + 10) % 255, (yy * 3 + 15) % 255), -1).astype(np.uint8)
    splits = {}
    spec = [('train', ['a', 'b']), ('val', ['c'])]
    if with_test:
        spec.append(('test', ['d']))
    for split, identifiers in spec:
        snapshot = project / (split + '.txt')
        snapshot.write_text('\n'.join(identifiers) + '\n')
        splits[split] = dict(
            snapshot_path=str(snapshot),
            count=len(identifiers),
            sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        )
        for ident in identifiers:
            Image.fromarray(rgb).save(data / (ident + '.png'))
            Image.fromarray(mask).save(data / (ident + '_mask.png'))
    manifest = project / 'manifest.json'
    manifest.write_text(
        json.dumps(dict(dataset_root=str(data), image_pattern='{id}.png', mask_pattern='{id}_mask.png', splits=splits))
    )
    config = json.loads((ROOT / 'configs/diffusion_seg_ordinal.json').read_text())
    config.update(
        manifest=str(manifest),
        device='cpu',
        epochs=2,
        warmup_epochs=0,
        cpu_threads=1,
        crop_size=64,
        stride=32,
        batch_size=2,
        workers=0,
        timesteps=8,
        sampling_steps=2,
        window_batch_size=1,
        val_every=1,
        photometric_gain=0.1,
        photometric_gamma=0.1,
        min_free_gib=0.0,
        max_run_gib=0.2,
    )
    return config


def legacy_dataset_item(config, split, epoch, index):
    """Exact copy of legacy SegmentationDataset.__getitem__ before Ch4 options."""
    source = MaterialDataset(config['manifest'], split, augment=False)
    training = split == 'train'
    source_index = index // config['crops_per_image'] if training else index
    rgb, mask, ident = source.raw(source_index)
    native_mask = mask.copy()
    generator = np.random.default_rng(np.random.SeedSequence([config['seed'], epoch, index]))
    if training:
        values = rgb.astype(np.float32) / 255
        gamma = np.exp(generator.uniform(-config['photometric_gamma'], config['photometric_gamma']))
        gain = 1 + generator.uniform(-config['photometric_gain'], config['photometric_gain'], 3)
        rgb = (np.clip(values ** gamma * gain, 0, 1) * 255 + 0.5).astype(np.uint8)
        turns = int(generator.integers(4))
        rgb, mask = np.rot90(rgb, turns), np.rot90(mask, turns)
        for axis in (0, 1):
            if generator.random() < 0.5:
                rgb, mask = np.flip(rgb, axis), np.flip(mask, axis)
    rgb, mask = experiment.pad_full_image(rgb, mask, config['crop_size'])
    thumbnail, soft = experiment.global_inputs(rgb, mask)
    height, width = mask.shape
    item = dict(thumb=thumbnail, soft=soft, id=ident)
    if not training:
        item.update(image=experiment.normalize(rgb), mask=torch.from_numpy(native_mask.astype(np.int64)))
        return item
    size = config['crop_size']
    thin = np.empty(0, dtype=np.int64)
    if generator.random() < config['thin_crop_fraction']:
        thin = np.flatnonzero(((mask == 1) | (mask == 2)).ravel()[::7]) * 7
    if len(thin):
        center_y, center_x = divmod(int(generator.choice(thin)), width)
        offset = generator.integers(-size // 4, size // 4 + 1, 2)
        top = int(np.clip(center_y - size // 2 + offset[0], 0, height - size))
        left = int(np.clip(center_x - size // 2 + offset[1], 0, width - size))
    else:
        top, left = int(generator.integers(height - size + 1)), int(generator.integers(width - size + 1))
    item.update(
        image=experiment.normalize(rgb[top : top + size, left : left + size]),
        mask=torch.from_numpy(mask[top : top + size, left : left + size].astype(np.int64)),
        box=torch.tensor([top / height, left / width, (top + size) / height, (left + size) / width], dtype=torch.float32),
    )
    return item


class Ch4OptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_tree_equal(self, first, second):
        if torch.is_tensor(first):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, np.ndarray):
            self.assertTrue(np.array_equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for k in first:
                self.assert_tree_equal(first[k], second[k])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for l, r in zip(first, second):
                self.assert_tree_equal(l, r)
        else:
            self.assertEqual(first, second)

    def test_validate_config_new_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            base = make_fixture(Path(directory))
            experiment.validate_config(base)
            for bad in [
                dict(base, photometric_vignette=-0.01),
                dict(base, photometric_vignette=0.51),
                dict(base, photometric_vignette=True),
                dict(base, photometric_vignette=float('nan')),
                dict(base, flip_mode='vertical'),
                dict(base, flip_mode=None),
                dict(base, ema_sigma_rel=0.0),
                dict(base, ema_sigma_rel=1.0),
                dict(base, ema_sigma_rel=True),
                dict(base, ema_sigma_rel=float('inf')),
            ]:
                with self.assertRaises(ValueError):
                    experiment.validate_config(bad)
            ok = dict(base, photometric_vignette=0.15, flip_mode='horizontal', ema_sigma_rel=0.1)
            experiment.validate_config(ok)
        # Validate production ch4 configs
        for name in ('ch4_g_onehot.json', 'ch4_g_ordinal.json'):
            cfg = json.loads((ROOT / 'configs' / name).read_text())
            experiment.validate_config(cfg)

    def test_default_augmentation_is_byte_identical_to_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            base = make_fixture(Path(directory))
            ds = experiment.SegmentationDataset(base, 'train')
            ds.epoch = 3
            for idx in range(len(ds)):
                ref = legacy_dataset_item(base, 'train', 3, idx)
                got = ds[idx]
                self.assert_tree_equal(got, ref)
            explicit_default = dict(base, photometric_vignette=0.0, flip_mode='both', ema_sigma_rel=None)
            ds2 = experiment.SegmentationDataset(explicit_default, 'train')
            ds2.epoch = 3
            for idx in range(len(ds2)):
                self.assert_tree_equal(ds2[idx], ds[idx])

    def test_vignette_and_horizontal_flip_match_glpflow_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_fixture(Path(directory))
            cfg.update(photometric_vignette=0.15, flip_mode='horizontal', thin_crop_fraction=0.0)
            ds = experiment.SegmentationDataset(cfg, 'train')
            ds.epoch = 2
            source = MaterialDataset(cfg['manifest'], 'train', augment=False)
            for idx in range(len(ds)):
                rgb_raw, mask_raw, ident = source.raw(idx // cfg['crops_per_image'])
                g = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 2, idx]))
                p_dict = dict(gain=cfg['photometric_gain'], gamma=cfg['photometric_gamma'], vignette=0.15)
                rgb_g = G.photometric(rgb_raw, g, p_dict)
                turns = int(g.integers(4))
                rgb_g, mask_g = np.rot90(rgb_g, turns), np.rot90(mask_raw, turns)
                if g.random() < 0.5:
                    rgb_g, mask_g = np.ascontiguousarray(rgb_g[:, ::-1]), np.ascontiguousarray(mask_g[:, ::-1])
                rgb_pad, mask_pad = experiment.pad_full_image(rgb_g, mask_g, cfg['crop_size'])
                thumb_g, soft_g = experiment.global_inputs(rgb_pad, mask_pad)
                # RNG next draw in SegmentationDataset when thin_crop_fraction==0 is uniform random vs top/left
                _ = g.random()
                top = int(g.integers(mask_pad.shape[0] - cfg['crop_size'] + 1))
                left = int(g.integers(mask_pad.shape[1] - cfg['crop_size'] + 1))
                sz = cfg['crop_size']
                expected = dict(
                    thumb=thumb_g,
                    soft=soft_g,
                    id=ident,
                    image=experiment.normalize(rgb_pad[top : top + sz, left : left + sz]),
                    mask=torch.from_numpy(mask_pad[top : top + sz, left : left + sz].astype(np.int64)),
                    box=torch.tensor(
                        [top / mask_pad.shape[0], left / mask_pad.shape[1], (top + sz) / mask_pad.shape[0], (left + sz) / mask_pad.shape[1]],
                        dtype=torch.float32,
                    ),
                )
                got = ds[idx]
                self.assert_tree_equal(got, expected)

    def test_power_ema_gamma_and_update_match_glpflow(self):
        for srel in (0.05, 0.1, 0.2):
            self.assertAlmostEqual(experiment.PowerEMA.solve(srel), G.PowerEMA.solve(srel), places=12)
        m1 = TinyDenoiser()
        m2 = TinyDenoiser()
        m2.load_state_dict(m1.state_dict())
        e1 = experiment.PowerEMA(m1, 0.1)
        e2 = G.PowerEMA(m2, 0.1)
        for step in range(5):
            with torch.no_grad():
                for p1, p2 in zip(m1.parameters(), m2.parameters()):
                    p1.add_(0.05 * (step + 1))
                    p2.add_(0.05 * (step + 1))
            e1.update(m1)
            e2.update(m2)
        self.assert_tree_equal(e1.state_dict(), e2.state_dict())

    def test_ema_resume_equivalence_and_default_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            cfg = make_fixture(project)
            cfg_ema = dict(cfg, photometric_vignette=0.15, flip_mode='horizontal', ema_sigma_rel=0.1)
            full, resumed = project / 'outputs/ema_full', project / 'outputs/ema_resumed'
            with patch.object(experiment, 'ConditionalMaskDenoiser', TinyDenoiser):
                experiment.train(cfg_ema, full, project, max_batches=2)
                experiment.train(cfg_ema, resumed, project, stop_after_epoch=1, max_batches=2)
                experiment.train(cfg_ema, resumed, project, resume=True, max_batches=2)
            ref = torch.load(full / 'last.pt', weights_only=False)
            rec = torch.load(resumed / 'last.pt', weights_only=False)
            self.assertIn('ema', ref)
            for key in ('model', 'ema', 'optimizer', 'scheduler', 'rng', 'epoch', 'global_step'):
                self.assert_tree_equal(ref[key], rec[key])
            self.assertEqual(ref['best']['epoch'], rec['best']['epoch'])
            self.assert_tree_equal(ref['best']['model'], rec['best']['model'])
            best_pt = torch.load(resumed / 'best.pt', weights_only=False)
            self.assert_tree_equal(best_pt['model'], rec['best']['model'])
            meta = json.loads((resumed / 'metadata.json').read_text())
            self.assertEqual(meta['ema'], dict(type='PowerEMA', sigma_rel=0.1))

            # Verify default ema=None produces no ema key
            plain = project / 'outputs/plain'
            with patch.object(experiment, 'ConditionalMaskDenoiser', TinyDenoiser):
                experiment.train(cfg, plain, project, max_batches=2)
            plain_last = torch.load(plain / 'last.pt', weights_only=False)
            self.assertNotIn('ema', plain_last)
            plain_meta = json.loads((plain / 'metadata.json').read_text())
            self.assertIs(plain_meta['ema'], False)

    def test_evaluate_overrides_and_test_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_fixture(Path(directory), with_test=True)
            with self.assertRaises(ValueError):
                experiment.SegmentationDataset(cfg, 'test')
            ds_test = experiment.SegmentationDataset(cfg, 'test', allow_test=True)
            self.assertEqual(len(ds_test), 1)
            model = TinyDenoiser()
            diffusion = GaussianMaskDiffusion(MaskStateCodec('ordinal'), cfg['timesteps'])
            res1 = experiment.evaluate(model, diffusion, cfg, split='val', sampling_steps=3, sampling_seed=3407, extra_tolerances=(1, 2, 4))
            res2 = experiment.evaluate(model, diffusion, cfg, split='val', sampling_steps=1, sampling_seed=3408, extra_tolerances=(1, 2, 4))
            self.assertEqual(cfg['sampling_steps'], 2)
            self.assertEqual(cfg['sampling_seed'], 3407)
            self.assertEqual(res1['images'][0]['sampling_steps'], 3)
            self.assertEqual(res2['images'][0]['sampling_steps'], 1)
            self.assertAlmostEqual(
                res1['metrics']['boundary_f1_foreground'],
                res1['metrics']['boundary_f1_foreground_tol2'],
                places=12,
            )
            self.assertIn('boundary_f1_foreground_tol1', res1['metrics'])
            self.assertIn('boundary_f1_foreground_tol4', res1['metrics'])
            res_test = experiment.evaluate(model, diffusion, cfg, split='test', allow_test=True)
            self.assertEqual(res_test['images'][0]['id'], 'd')


if __name__ == '__main__':
    unittest.main()

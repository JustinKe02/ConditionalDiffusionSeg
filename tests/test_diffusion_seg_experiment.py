"""Synthetic CPU regressions for geometry, window noise and epoch recovery."""

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
from conditional_diffusion import diffusion_seg_experiment as experiment
from conditional_diffusion.data import MEAN, STD
from conditional_diffusion.mask_diffusion import GaussianMaskDiffusion, MaskStateCodec


class TinyDenoiser(torch.nn.Module):
    """Cheap model with the production condition and denoiser contracts."""

    def __init__(self, state_channels=3, timesteps=8, **kwargs):
        super().__init__()
        self.head = torch.nn.Conv2d(3, 4, 1)
        self.prior = torch.nn.Conv2d(3, 4, 1)
        self.state_scale = torch.nn.Parameter(torch.tensor(.1))
        self.timesteps = timesteps
        self.states = []
        self.global_calls = 0

    def encode_global(self, thumbnails):
        self.global_calls += 1
        return [(thumbnail.flatten(1).T, self.prior(F.avg_pool2d(thumbnail[None], 4))[0], (1, 1))
                for thumbnail in thumbnails]

    def encode_condition(self, image, boxes, contexts):
        return image, torch.stack([context[1].mean((1, 2)) for context in contexts])

    def denoise(self, condition, state, times):
        self.states.append(state.detach().cpu().clone())
        image, prior = condition
        return self.head(image) + prior[:, :, None, None] + self.state_scale * state.mean(1, keepdim=True)


def fixture(project, height=79, width=93):
    data = project / 'data'
    data.mkdir()
    yy, xx = np.indices((height, width))
    mask = ((yy >= height // 2) * 2 + (xx >= width // 2)).astype(np.uint8)
    rgb = np.stack((mask * 60, xx.astype(np.uint8), yy.astype(np.uint8)), -1)
    splits = {}
    for split, identifiers in (('train', ['a', 'b']), ('val', ['c'])):
        snapshot = project / (split + '.txt')
        snapshot.write_text('\n'.join(identifiers) + '\n')
        splits[split] = dict(snapshot_path=str(snapshot), count=len(identifiers),
                             sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
        for ident in identifiers:
            Image.fromarray(rgb).save(data / (ident + '.png'))
            Image.fromarray(mask).save(data / (ident + '_mask.png'))
    manifest = project / 'manifest.json'
    manifest.write_text(json.dumps(dict(dataset_root=str(data), image_pattern='{id}.png',
                                        mask_pattern='{id}_mask.png', splits=splits)))
    config = json.loads((ROOT / 'configs/diffusion_seg_ordinal.json').read_text())
    config.update(manifest=str(manifest), device='cpu', epochs=2, warmup_epochs=0, cpu_threads=1,
                  crop_size=64, stride=32, batch_size=2, workers=0, timesteps=8, sampling_steps=2,
                  window_batch_size=1, val_every=1, photometric_gain=0., photometric_gamma=0.,
                  min_free_gib=0., max_run_gib=.1)
    return config


class ExperimentTests(unittest.TestCase):
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
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_dataset_crop_alignment_padding_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            dataset = experiment.SegmentationDataset(config)
            dataset.epoch = 2
            item = dataset[0]
            self.assertEqual(len(dataset), 4)
            self.assert_tree_equal(item, dataset[0])
            reconstructed = ((item['image'] * STD + MEAN) * 255).round().long()
            self.assertTrue(torch.equal(reconstructed[0] // 60, item['mask']))
            self.assertTrue(bool(((item['box'] >= 0) & (item['box'] <= 1)).all()))
            self.assertEqual(tuple(item['thumb'].shape), (3, 32, 32))
            torch.testing.assert_close(item['soft'].sum(0), torch.ones(8, 8))
            validation = experiment.SegmentationDataset(config, 'val')[0]
            self.assertEqual(tuple(validation['image'].shape), (3, 128, 128))
            self.assertEqual(tuple(validation['mask'].shape), (79, 93))
            with self.assertRaises(ValueError):
                experiment.SegmentationDataset(config, 'test')

    def test_window_noise_is_shared_global_cached_and_native_size_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            config['sampling_steps'] = 1
            item = experiment.SegmentationDataset(config, 'val')[0]
            model = TinyDenoiser()
            diffusion = GaussianMaskDiffusion(MaskStateCodec(), 8)
            prediction, timing = experiment.predict_image(model, diffusion, item['image'], item['thumb'],
                                                          item['id'], config, item['mask'].shape)
            self.assertEqual(prediction.shape, (79, 93))
            self.assertEqual(timing['nfe'], 9)
            self.assertEqual(model.global_calls, 1)
            self.assertTrue(torch.equal(model.states[0][:, :, :, 32:], model.states[1][:, :, :, :32]))
            model.states.clear()
            config['method'] = 'direct'
            _, timing = experiment.predict_image(model, diffusion, item['image'], item['thumb'], item['id'], config)
            self.assertEqual(timing['nfe'], 9)
            self.assertTrue(all(bool((state == 0).all()) for state in model.states))

    def test_epoch_resume_restores_optimizer_scheduler_rng_and_best_export(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            config = fixture(project)
            full, resumed = project / 'outputs/full', project / 'outputs/resumed'
            with patch.object(experiment, 'ConditionalMaskDenoiser', TinyDenoiser):
                experiment.train(config, full, project)
                experiment.train(config, resumed, project, stop_after_epoch=1)
                (resumed / 'best.pt').write_bytes(b'interrupted export')
                experiment.train(config, resumed, project, resume=True)
            reference = torch.load(full / 'last.pt', weights_only=False)
            recovered = torch.load(resumed / 'last.pt', weights_only=False)
            for key in ('model', 'optimizer', 'scheduler', 'rng', 'epoch', 'global_step', 'best'):
                if key == 'best':
                    self.assert_tree_equal(reference[key]['model'], recovered[key]['model'])
                    self.assertEqual(reference[key]['epoch'], recovered[key]['epoch'])
                else:
                    self.assert_tree_equal(reference[key], recovered[key])
            selected = torch.load(resumed / 'best.pt', weights_only=False)
            self.assert_tree_equal(selected['model'], recovered['best']['model'])
            with self.assertRaises(FileExistsError):
                experiment.train(config, resumed, project)
            changed = dict(config, lr=1e-3)
            with self.assertRaises(ValueError):
                experiment.train(changed, resumed, project, resume=True)

    def test_lock_and_storage_budget_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            config = fixture(project)
            run = project / 'outputs/locked'
            with experiment.run_lock(run):
                with self.assertRaises(RuntimeError):
                    with experiment.run_lock(run):
                        self.fail('Concurrent owner acquired lock')
            run.mkdir()
            guard = experiment.RunStorage(run, dict(config, max_run_gib=1e-9))
            with self.assertRaises(RuntimeError):
                experiment.atomic_save(run / 'state.json', {'ok': True}, guard)
            self.assertFalse((run / 'state.json').exists())

    def test_tiny_image_padding_and_direct_smoke_use_production_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory), height=21, width=37)
            config['method'] = 'direct'
            item = experiment.SegmentationDataset(config)[0]
            self.assertEqual(tuple(item['thumb'].shape), (3, 32, 32))
            model = TinyDenoiser()
            with patch.object(experiment, 'ConditionalMaskDenoiser', return_value=model), \
                    patch.object(GaussianMaskDiffusion, 'sample', side_effect=AssertionError('Direct must not sample')):
                result = experiment.smoke(config)
            self.assertEqual(result['sampling_steps'], 1)
            self.assertEqual(result['method'], 'direct')
            self.assertTrue(all(bool((state == 0).all()) for state in model.states))

    def test_material_errors_use_original_class_order_and_all_pixels(self):
        matrix = np.array([[10, 1, 2, 3], [4, 11, 5, 6], [7, 8, 12, 9], [1, 2, 3, 13]])
        metrics = experiment.summarize_material(matrix)
        self.assertAlmostEqual(metrics['monolayer_fewlayer_error_rate'], 13 / matrix.sum())
        self.assertAlmostEqual(metrics['cross_level_error_rate'], (2 + 3 + 6 + 7 + 1 + 2) / matrix.sum())
        self.assertAlmostEqual(metrics['thin_mean_iou'], (metrics['iou_monolayer'] + metrics['iou_fewlayer']) / 2)

    def test_failed_initialization_can_be_explicitly_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            config = fixture(project)
            run = project / 'outputs/init_failed'
            with patch.object(experiment, 'ConditionalMaskDenoiser', side_effect=RuntimeError('initialization failed')):
                with self.assertRaises(RuntimeError):
                    experiment.train(config, run, project)
            self.assertFalse((run / 'last.pt').exists())
            with patch.object(experiment, 'ConditionalMaskDenoiser', TinyDenoiser):
                result = experiment.train(config, run, project, resume=True, stop_after_epoch=1)
            self.assertEqual(result['epoch'], 1)

    def test_changed_source_cannot_be_committed_under_old_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            config = fixture(project)
            run = project / 'outputs/source_changed'
            source = project / 'src'
            source.mkdir()
            observed = source / 'observed.py'
            observed.write_text('version = 1\n')
            original_step = experiment.train_step

            def changing_step(*args):
                result = original_step(*args)
                observed.write_text('version = 2\n')
                return result

            with patch.object(experiment, 'ConditionalMaskDenoiser', TinyDenoiser), \
                    patch.object(experiment, 'train_step', side_effect=changing_step):
                with self.assertRaises(ValueError):
                    experiment.train(config, run, project, stop_after_epoch=1)
            state = torch.load(run / 'last.pt', weights_only=False)
            self.assertEqual(state['epoch'], 0)


if __name__ == '__main__':
    unittest.main()

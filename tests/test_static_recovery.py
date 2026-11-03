"""Temporary-directory CPU tests for actual static trainer recovery branches.

The model/data/logger are fixtures; real checkpoint serialization and trainer
control flow are retained. Every attempted loader stops before a training batch.
"""
from contextlib import ExitStack, contextmanager
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion import static_experiment as experiment
from conditional_diffusion.runtime import rng_state


class BeforeFirstBatch(RuntimeError):
    pass


class FakeDataset:
    def __init__(self, manifest, split, **kwargs):
        self.split = split

    def __len__(self):
        return 120 if self.split == 'train' else 25

    def __getitem__(self, index):
        raise AssertionError('A recovery-only test must not load a training/evaluation sample')


class TinyModel(torch.nn.Module):
    def __init__(self, value=.125):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((4, 3), value))

    def forward(self, *args, **kwargs):
        raise AssertionError('A recovery-only test must not execute the model')


class StaticRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self):
        config = json.loads((ROOT / 'outputs/P1/direct/seed_42/p1_cosine_20261017/config.json').read_text())
        config.update(mode='S0', device='cpu', cpu_threads=2, workers=0,
                      window_batch_size=1, sampling_steps=5)
        return config

    @contextmanager
    def environment(self, loader_callback=None, model_value=.125):
        models = []
        schedulers = []
        actual_scheduler = experiment.scheduler_for
        def build(config):
            model = TinyModel(model_value)
            models.append(model)
            return model
        def scheduler(optimizer, config):
            result = actual_scheduler(optimizer, config)
            schedulers.append(result)
            return result
        manifest = {'splits': {k: {'sha256': 'test-fixture-' + k} for k in ['train', 'val', 'test']}}
        if loader_callback is None:
            loader_callback = MagicMock(side_effect=BeforeFirstBatch('controlled stop before first batch'))
        with ExitStack() as stack:
            stack.enter_context(patch.object(experiment, 'read_manifest', return_value=manifest))
            stack.enter_context(patch.object(experiment, 'MaterialDataset', FakeDataset))
            stack.enter_context(patch.object(experiment, 'build', side_effect=build))
            stack.enter_context(patch.object(experiment, 'scheduler_for', side_effect=scheduler))
            guard = stack.enter_context(patch.object(experiment, 'StorageGuard'))
            guard.return_value.check.return_value = {}
            logger = stack.enter_context(patch.object(experiment, 'ExperimentLogger'))
            loader = stack.enter_context(patch.object(experiment, 'loader_for', side_effect=loader_callback))
            train_step = stack.enter_context(patch.object(experiment, 'train_step', side_effect=AssertionError('No optimizer update allowed')))
            evaluation = stack.enter_context(patch.object(experiment, 'evaluate_static', side_effect=AssertionError('No evaluation allowed')))
            stack.enter_context(patch('torch.cuda.is_available', return_value=False))
            stack.enter_context(patch('torch.cuda.get_device_name', return_value='CPU-only mock; no CUDA access'))
            stack.enter_context(patch('torch.cuda.reset_peak_memory_stats', side_effect=AssertionError('No CUDA access allowed')))
            yield {'models': models, 'schedulers': schedulers, 'loader': loader, 'logger': logger,
                   'train_step': train_step, 'evaluation': evaluation}
            train_step.assert_not_called()
            evaluation.assert_not_called()

    def fixture_last(self, config, epoch, step):
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = experiment.scheduler_for(optimizer, config)
        return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'rng': rng_state(),
                'config': config, 'source_sha256': {}, 'epoch': epoch,
                'global_step': step, 'best_miou': .2}

    def canonical_best(self, config):
        matrix = np.diag([11, 7, 5, 3]).astype(np.int64)
        matrices = {'ids': np.array(['recovery_fixture']), 'global_matrix': matrix,
                    'per_image': matrix[None], 'boundary_counts': np.zeros((4, 4), dtype=np.int64)}
        return {'model': TinyModel().state_dict(), 'config': config, 'source_sha256': {},
                'epoch': 30, 'global_step': 1800,
                'validation_metrics': {'miou_4class': .42, 'image_count': 25},
                'validation_matrices': matrices}

    def assert_sidecars_equal(self, run, payload):
        saved = json.loads((run / 'best_val_metrics.json').read_text())
        self.assertEqual(saved, {'epoch': payload['epoch'], 'metrics': payload['validation_metrics']})
        with np.load(run / 'best_val_confusion.npz', allow_pickle=False) as saved_matrix:
            self.assertEqual(set(saved_matrix.files), set(payload['validation_matrices']))
            for key, expected in payload['validation_matrices'].items():
                self.assertTrue(np.array_equal(saved_matrix[key], expected), key)
        self.assertFalse((run / 'best_val_confusion.recovery.npz').exists())

    def test_missing_and_corrupt_best_sidecars_rebuilt_from_canonical_payload(self):
        config = self.config()
        payload = self.canonical_best(config)
        for damaged in ['missing', 'corrupt']:
            with self.subTest(state=damaged), tempfile.TemporaryDirectory() as folder:
                run = Path(folder)
                if damaged == 'corrupt':
                    (run / 'best_val_metrics.json').write_text('not-json')
                    (run / 'best_val_confusion.npz').write_bytes(b'truncated-array')
                experiment.restore_best_sidecars(run, payload)
                self.assert_sidecars_equal(run, payload)

    def test_epoch40_last_restores_canonical_best_and_completes_without_update(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as folder, self.environment() as env:
            project, run = Path(folder), Path(folder) / 'run'
            (run / 'checkpoints').mkdir(parents=True)
            torch.save(self.fixture_last(config, 40, 2400), run / 'checkpoints/last.pt')
            payload = self.canonical_best(config)
            torch.save(payload, run / 'checkpoints/best.pt')
            (run / 'best_val_metrics.json').write_text('{"epoch":40,"metrics":{"miou_4class":0.999}}')
            (run / 'best_val_confusion.npz').write_bytes(b'corrupted-sidecar')
            old_last = hashlib.sha256((run / 'checkpoints/last.pt').read_bytes()).hexdigest()
            experiment.train_static(config, project, run, {}, resume=True)
            self.assertEqual(json.loads((run / 'status.json').read_text()),
                             {'status': 'completed', 'last_completed_epoch': 40, 'global_step': 2400})
            self.assert_sidecars_equal(run, payload)
            self.assertEqual(old_last, hashlib.sha256((run / 'checkpoints/last.pt').read_bytes()).hexdigest())
            env['loader'].assert_not_called()
            env['logger'].assert_not_called()

    def test_epoch0_checkpoint_written_before_first_loader_and_fully_restorable(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as folder:
            project, run = Path(folder), Path(folder) / 'run'
            with self.environment() as initial:
                with self.assertRaises(BeforeFirstBatch):
                    experiment.train_static(config, project, run, {}, resume=False)
                initial['logger'].return_value.close.assert_called_once()
            last = run / 'checkpoints/last.pt'
            self.assertTrue(last.exists())
            state = torch.load(last, map_location='cpu', weights_only=False)
            self.assertEqual((state['epoch'], state['global_step']), (0, 0))
            for key in ['model', 'optimizer', 'scheduler', 'rng', 'config', 'source_sha256', 'best_miou']:
                self.assertIn(key, state)
            self.assertEqual(state['config'], config)
            self.assertEqual(state['optimizer']['state'], {})
            self.assertTrue(torch.equal(state['model']['weight'], torch.full((4, 3), .125)))
            original_hash = hashlib.sha256(last.read_bytes()).hexdigest()
            # Force a different freshly built model, proving load_state_dict is used.
            with self.environment(model_value=-7.) as resumed:
                with self.assertRaises(BeforeFirstBatch):
                    experiment.train_static(config, project, run, {}, resume=True)
                self.assertTrue(torch.equal(resumed['models'][0].weight.detach(), state['model']['weight']))
                self.assertEqual(resumed['schedulers'][0].state_dict(), state['scheduler'])
                self.assertEqual(resumed['schedulers'][0].optimizer.state_dict(), state['optimizer'])
                restored = rng_state()
                self.assertTrue(torch.equal(restored['torch'], state['rng']['torch']))
                self.assertEqual(restored['python'], state['rng']['python'])
                self.assertEqual(restored['numpy'][0], state['rng']['numpy'][0])
                self.assertTrue(np.array_equal(restored['numpy'][1], state['rng']['numpy'][1]))
                self.assertEqual(restored['numpy'][2:], state['rng']['numpy'][2:])
                self.assertEqual(resumed['logger'].call_args.args[2:], (0, 0))
                resumed['logger'].return_value.close.assert_called_once()
            self.assertEqual(original_hash, hashlib.sha256(last.read_bytes()).hexdigest())
            status = json.loads((run / 'status.json').read_text())
            self.assertEqual(status['status'], 'interrupted')
            self.assertEqual(status['global_step'], 0)

    def test_missing_last_initialization_directory_is_preserved_by_archiving(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as folder, self.environment() as env:
            project, run = Path(folder), Path(folder) / 'run'
            (run / 'checkpoints').mkdir(parents=True)
            config_bytes = json.dumps(config).encode()
            (run / 'config.json').write_bytes(config_bytes)
            (run / 'keep_initial_failure.txt').write_bytes(b'preserve this failed initialization')
            with self.assertRaises(BeforeFirstBatch):
                experiment.train_static(config, project, run, {}, resume=True)
            archived = list(project.glob('run_incomplete_start_*'))
            self.assertEqual(len(archived), 1)
            self.assertEqual((archived[0] / 'config.json').read_bytes(), config_bytes)
            self.assertEqual((archived[0] / 'keep_initial_failure.txt').read_bytes(), b'preserve this failed initialization')
            self.assertFalse((run / 'keep_initial_failure.txt').exists())
            self.assertTrue((run / 'checkpoints/last.pt').exists())
            state = torch.load(run / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
            self.assertEqual((state['epoch'], state['global_step']), (0, 0))
            self.assertEqual(env['logger'].call_args.args[2:], (None, None))

    def test_missing_last_with_best_or_update_log_refuses_automatic_archiving(self):
        config = self.config()
        for artifact in ['checkpoints/best.pt', 'metrics_steps.csv']:
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as folder, self.environment() as env:
                project, run = Path(folder), Path(folder) / 'run'
                (run / 'checkpoints').mkdir(parents=True)
                (run / artifact).write_bytes(b'must not discard evidence of work')
                with self.assertRaises(AssertionError):
                    experiment.train_static(config, project, run, {}, resume=True)
                self.assertEqual(list(project.glob('run_incomplete_start_*')), [])
                self.assertEqual((run / artifact).read_bytes(), b'must not discard evidence of work')
                env['loader'].assert_not_called()


if __name__ == '__main__':
    unittest.main()

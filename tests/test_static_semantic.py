"""CPU contracts for the registered static-RGB S0/S1/S2 platform.

Synthetic 32/64 crops only: these checks are not an effect-training experiment.
"""
import copy
import io
import math
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.objectives import losses
from conditional_diffusion.recycling import mixed_recycling_input
from conditional_diffusion.runtime import rng_state, restore_rng
from conditional_diffusion.static_semantic import StaticSemanticModel, sample_cached
from conditional_diffusion.static_experiment import scheduler_for as platform_scheduler_for, train_step


def seed_cpu(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def scheduler_for(optimizer):
    """Exercise the actual new workflow's registered scheduler, not a copy."""
    return platform_scheduler_for(optimizer, {'warmup_epochs': 5, 'epochs': 40,
                                            'min_lr': 1e-6, 'lr': 2e-4})


def synthetic_update(model, optimizer, scheduler):
    # All three host RNG streams are used and checked after restoring a checkpoint.
    crop = 32
    scale = .75 + .5 * random.random()
    image = torch.randn(2, 3, crop, crop) * scale
    labels = torch.from_numpy(np.random.randint(0, 4, (2, crop, crop))).long()
    d = GaussianDiffusion(1000)
    terms, _, _ = train_step(model, d, optimizer, image, labels,
                             {'mode': model.mode, 'boundary': False, 'grad_clip': 1.})
    scheduler.step()
    return terms['loss_total']


class StaticSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_tree_equal(self, a, b, path='root'):
        self.assertIs(type(a), type(b), path)
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b), path)
        elif isinstance(a, np.ndarray):
            self.assertTrue(np.array_equal(a, b), path)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys(), path)
            for key in a:
                self.assert_tree_equal(a[key], b[key], path + '/' + str(key))
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b), path)
            for i, (x, y) in enumerate(zip(a, b)):
                self.assert_tree_equal(x, y, path + '/' + str(i))
        else:
            self.assertEqual(a, b, path)

    def test_shared_initialization_and_s0_has_no_idle_state_modules(self):
        models = {mode: StaticSemanticModel(mode, seed=42) for mode in ['S0', 'S1', 'S2']}
        for name in ['rgb_encoder', 'decoder', 'head']:
            expected = getattr(models['S0'], name).state_dict()
            for mode in ['S1', 'S2']:
                self.assert_tree_equal(expected, getattr(models[mode], name).state_dict(), mode + '/' + name)
        for name in ['state_encoder', 'state_projection', 'time_mlp']:
            self.assert_tree_equal(getattr(models['S1'], name).state_dict(),
                                   getattr(models['S2'], name).state_dict(), name)
        self.assertFalse(any('state' in name or 'time' in name for name, _ in models['S0'].named_parameters()))
        self.assertLess(sum(p.numel() for p in models['S0'].parameters()),
                        sum(p.numel() for p in models['S1'].parameters()))
        self.assertEqual(sum(p.numel() for p in models['S1'].parameters()),
                         sum(p.numel() for p in models['S2'].parameters()))

    def test_initialization_is_rng_isolated_and_seed_sensitive(self):
        seed_cpu(130)
        before = torch.get_rng_state().clone()
        m = StaticSemanticModel('S2', seed=42)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        same = StaticSemanticModel('S0', seed=42)
        different = StaticSemanticModel('S0', seed=43)
        self.assert_tree_equal(m.rgb_encoder.state_dict(), same.rgb_encoder.state_dict())
        self.assertTrue(any(not torch.equal(v, different.rgb_encoder.state_dict()[k])
                            for k, v in same.rgb_encoder.state_dict().items()))

    def test_rgb_multiscale_shapes_and_64_pixel_output(self):
        model = StaticSemanticModel('S0')
        image = torch.randn(2, 3, 64, 64)
        features = model.encode_rgb(image)
        self.assertEqual([tuple(x.shape) for x in features],
                         [(2, 32, 64, 64), (2, 64, 32, 32), (2, 128, 16, 16), (2, 256, 8, 8)])
        self.assertTrue(torch.equal(model.forward_with_rgb(features), model(image)))
        self.assertEqual(tuple(model(image).shape), (2, 4, 64, 64))

    def test_s0_every_parameter_is_effective(self):
        model = StaticSemanticModel('S0')
        image = torch.randn(2, 3, 32, 32)
        mask = torch.randint(4, (2, 32, 32))
        loss, _ = losses(model(image), mask, {'boundary': False})
        loss.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(float(p.grad.abs().sum()), 0, name)

    def test_s1_uniform_state_fixed_grid_full_gradient_and_one_rgb_encoding(self):
        model = StaticSemanticModel('S1')
        image = torch.randn(2, 3, 32, 32)
        calls = []
        original = model.forward_with_rgb
        def capture(features, state=None, t=None):
            result = original(features, state, t)
            result.retain_grad()
            calls.append((state, t.detach().clone(), result))
            return result
        with patch.object(model, 'encode_rgb', wraps=model.encode_rgb) as rgb, \
                patch.object(model, 'forward_with_rgb', side_effect=capture):
            out = model(image)
        self.assertEqual(rgb.call_count, 1)
        self.assertEqual(len(calls), 5)
        self.assertEqual([int(t[0]) for _, t, _ in calls], [999, 749, 500, 250, 0])
        self.assertTrue(torch.equal(calls[0][0], torch.full_like(calls[0][0], -.5)))
        for previous, current in zip(calls, calls[1:]):
            self.assertTrue(torch.equal(current[0], 2 * previous[2].softmax(1) - 1))
            self.assertTrue(current[0].requires_grad)
        out.square().mean().backward()
        for i, (_, _, prediction) in enumerate(calls):
            self.assertIsNotNone(prediction.grad, 'step ' + str(i))
            self.assertGreater(float(prediction.grad.abs().sum()), 0, 'step ' + str(i))
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.rgb_encoder.parameters()), 0)

    def test_s2_cached_and_uncached_ddim_states_and_outputs_match(self):
        seed_cpu(42)
        model = StaticSemanticModel('S2').eval()
        d = GaussianDiffusion(1000)
        image, noise = torch.randn(2, 3, 32, 32), torch.randn(2, 4, 32, 32)
        initial_noise = noise.clone()
        original = model.forward_with_rgb
        records = []
        def capture(features, state=None, t=None):
            output = original(features, state, t)
            records.append((state.detach().clone(), t.detach().clone(), output.detach().clone()))
            return output
        with patch.object(model, 'encode_rgb', wraps=model.encode_rgb) as rgb, \
                patch.object(model, 'forward_with_rgb', side_effect=capture):
            cached = sample_cached(model, d, image, noise, steps=5)
            cached_calls = records[:]
            records.clear()
            self.assertEqual(rgb.call_count, 1)
            rgb.reset_mock()
            uncached = d.sample(model, image, noise, steps=5)
            self.assertEqual(rgb.call_count, 5)
        self.assertTrue(torch.equal(cached, uncached))
        self.assertTrue(torch.equal(noise, initial_noise))
        self.assertFalse(cached.requires_grad)
        self.assertEqual(len(cached_calls), 5)
        self.assertEqual([int(t[0]) for _, t, _ in cached_calls], [999, 749, 500, 250, 0])
        for i, (cached_step, plain_step) in enumerate(zip(cached_calls, records)):
            self.assert_tree_equal(cached_step, plain_step, 'DDIM step ' + str(i))
            if i < 4:
                xt, t, predicted = cached_step
                x0 = predicted.clamp(-1, 1)
                alpha, next_alpha = d.abar[t[0]], d.abar[cached_calls[i + 1][1][0]]
                eps = (xt - alpha.sqrt() * x0) / (1 - alpha).sqrt()
                independent_next = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * eps
                self.assertTrue(torch.equal(independent_next, cached_calls[i + 1][0]))

    def test_s2_state_time_and_cached_rgb_receive_effective_gradients(self):
        model = StaticSemanticModel('S2')
        image = torch.randn(2, 3, 32, 32)
        state = torch.randn(2, 4, 32, 32, requires_grad=True)
        t = torch.tensor([123., 987.], requires_grad=True)
        features = model.encode_rgb(image)
        prediction = model.forward_with_rgb(features, state, t)
        prediction.square().mean().backward()
        for tensor in [state, t]:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0)
        for name in ['rgb_encoder', 'decoder', 'head', 'state_encoder', 'state_projection', 'time_mlp']:
            params = list(getattr(model, name).parameters())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params), name)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in params), 0, name)
        with torch.no_grad():
            original = model.forward_with_rgb(features, state, t)
            different_state = model.forward_with_rgb(features, state + .25, t)
            different_time = model.forward_with_rgb(features, state, t + 10)
        self.assertGreater(float((original - different_state).abs().max()), 0)
        self.assertGreater(float((original - different_time).abs().max()), 0)

    def test_existing_recycling_with_s2_uses_independent_detached_teacher_noise(self):
        seed_cpu(14)
        model = StaticSemanticModel('S2')
        d = GaussianDiffusion(1000)
        image = torch.randn(4, 3, 32, 32)
        x0 = d.encode(torch.randint(4, (4, 32, 32)))
        original_x0 = x0.clone()
        t = d.training_timesteps(4, 'high_noise_stratified', 900)
        noise = torch.randn_like(x0)
        calls = []
        def hook(module, args, output):
            calls.append((args[1].clone(), args[2].clone(), torch.is_grad_enabled(), output.detach().clone()))
        handle = model.register_forward_hook(hook)
        try:
            with patch.object(model, 'encode_rgb', wraps=model.encode_rgb) as rgb:
                noisy, selected = mixed_recycling_input(model, d, image, x0, t, noise)
                self.assertEqual(rgb.call_count, 1)
        finally:
            handle.remove()
        self.assertEqual(selected.numel(), 2)
        self.assertEqual(selected.unique().numel(), 2)
        teacher_noise, terminal, grad_enabled, prediction = calls[0]
        self.assertFalse(grad_enabled)
        self.assertTrue((terminal == 999).all())
        self.assertFalse(torch.equal(teacher_noise, noise[selected]))
        self.assertFalse(torch.equal(teacher_noise, x0[selected]))
        self.assertFalse(noisy.requires_grad)
        self.assertTrue(torch.equal(x0, original_x0))
        expected_source = x0.clone()
        expected_source[selected] = prediction.clamp(-1, 1)
        self.assertTrue(torch.equal(noisy, d.q_sample(expected_source, t, noise)))
        loss, _ = losses(model(image, noisy, t), original_x0.argmax(1), {'boundary': False}, original_x0)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_complete_adam_scheduler_rng_checkpoint_resume_all_modes(self):
        # Mask the CUDA availability query only for runtime RNG serialization.
        # Models, batches and optimizer states remain on CPU throughout.
        with patch('torch.cuda.is_available', return_value=False):
            for mode in ['S0', 'S1', 'S2']:
                with self.subTest(mode=mode):
                    seed_cpu(3407)
                    model = StaticSemanticModel(mode)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
                    scheduler = scheduler_for(optimizer)
                    synthetic_update(model, optimizer, scheduler)
                    checkpoint = io.BytesIO()
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(), 'rng': rng_state(),
                                'epoch': 1, 'global_step': 1}, checkpoint)
                    continuous_losses = [synthetic_update(model, optimizer, scheduler) for _ in range(2)]
                    expected_model = copy.deepcopy(model.state_dict())
                    expected_optimizer = copy.deepcopy(optimizer.state_dict())
                    expected_scheduler = copy.deepcopy(scheduler.state_dict())
                    expected_rng = rng_state()
                    del model, optimizer, scheduler
                    checkpoint.seek(0)
                    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    model = StaticSemanticModel(mode)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
                    scheduler = scheduler_for(optimizer)
                    model.load_state_dict(state['model'])
                    optimizer.load_state_dict(state['optimizer'])
                    scheduler.load_state_dict(state['scheduler'])
                    restore_rng(state['rng'])
                    resumed_losses = [synthetic_update(model, optimizer, scheduler) for _ in range(2)]
                    self.assertEqual(continuous_losses, resumed_losses)
                    self.assert_tree_equal(expected_model, model.state_dict(), mode + '/model')
                    self.assert_tree_equal(expected_optimizer, optimizer.state_dict(), mode + '/AdamW')
                    self.assert_tree_equal(expected_scheduler, scheduler.state_dict(), mode + '/scheduler')
                    self.assert_tree_equal(expected_rng, rng_state(), mode + '/RNG')
                    del model, optimizer, scheduler


if __name__ == '__main__':
    unittest.main()

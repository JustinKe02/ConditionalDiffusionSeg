"""CPU contracts for matched single-head bounded-state P0/P1 controls.

All inputs are synthetic tiny crops. No effect training or dataset scoring is
performed; optimizer calls only test deterministic checkpoint restoration.
"""
import copy
import io
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion import static_experiment as shared
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.objectives import losses
from conditional_diffusion.recycling import mixed_recycling_input
from conditional_diffusion.runtime import rng_state, restore_rng
from conditional_diffusion.static_semantic import StaticSemanticModel
from conditional_diffusion.state_objective import (StateObjectiveModel, bounded_state,
                                                  sample_scores, bind_experiment)


def seed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def config(mapping):
    return {'mode': 'S2', 'state_mapping': mapping, 'seed': 42, 'device': 'cpu',
            'boundary': False, 'grad_clip': 1., 'lr': 2e-4, 'min_lr': 1e-6,
            'warmup_epochs': 5, 'epochs': 40}


class DictTeacher(torch.nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping
        self.logits = torch.nn.Parameter(torch.tensor([2., -2., 1., -1.])[None, :, None, None])
        self.calls = []

    def forward(self, image, state, t):
        scores = self.logits.expand_as(state)
        result = dict(logits=scores, x0=bounded_state(scores, self.mapping))
        self.calls.append({'noise': state.detach().clone(), 't': t.detach().clone(),
                           'grad_enabled': torch.is_grad_enabled(),
                           'logits': scores.detach().clone(), 'x0': result['x0'].detach().clone()})
        return result


class StateObjectiveTests(unittest.TestCase):
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
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b), path)
            for i, (x, y) in enumerate(zip(a, b)):
                self.assert_tree_equal(x, y, path + '/' + str(i))
        else:
            self.assertEqual(a, b, path)

    def test_exact_s2_initial_weights_parameter_count_and_single_head(self):
        original = StaticSemanticModel('S2', seed=42)
        for mapping in ['tanh', 'simplex']:
            with self.subTest(mapping=mapping):
                model = StateObjectiveModel(mapping, seed=42)
                self.assert_tree_equal(original.state_dict(), model.state_dict())
                self.assertEqual(sum(p.numel() for p in original.parameters()), 5_939_156)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 5_939_156)
                self.assertEqual(list(dict(original.named_parameters())), list(dict(model.named_parameters())))
                self.assertEqual(list(model.head.children())[-1].out_channels, 4)
                self.assertFalse(any('denoise' in name or 'classifier' in name for name in model.state_dict()))

    def test_bounded_ranges_simplex_channel_sum_and_shared_raw_logits(self):
        seed_cpu(42)
        image, state, t = torch.randn(2, 3, 32, 32), torch.randn(2, 4, 32, 32), torch.tensor([134, 978])
        outputs = {}
        for mapping in ['tanh', 'simplex']:
            model = StateObjectiveModel(mapping)
            output = model(image, state, t)
            self.assertEqual(set(output), {'logits', 'x0'})
            self.assertEqual(output['x0'].shape, (2, 4, 32, 32))
            self.assertTrue(torch.isfinite(output['x0']).all())
            self.assertTrue((output['x0'] >= -1).all() and (output['x0'] <= 1).all())
            expected = output['logits'].tanh() if mapping == 'tanh' else 2*output['logits'].softmax(1)-1
            self.assertTrue(torch.equal(output['x0'], expected))
            outputs[mapping] = output
        self.assertTrue(torch.equal(outputs['tanh']['logits'], outputs['simplex']['logits']))
        summed = outputs['simplex']['x0'].sum(1)
        self.assertTrue(torch.allclose(summed, torch.full_like(summed, -2), atol=5e-7, rtol=0))
        self.assertFalse(torch.equal(outputs['tanh']['x0'], outputs['simplex']['x0']))

    def test_cached_sampler_matches_existing_dict_ddim_and_raw_final_readout(self):
        for mapping in ['tanh', 'simplex']:
            with self.subTest(mapping=mapping):
                model = StateObjectiveModel(mapping).eval()
                # Raw scores outside [-1,1] make clipped/denoising readout bugs observable.
                with torch.no_grad():
                    model.head[-1].bias.copy_(torch.tensor([3., -2., 1., -.5]))
                image, noise = torch.randn(2, 3, 32, 32), torch.randn(2, 4, 32, 32)
                input_noise = noise.clone()
                diffusion = GaussianDiffusion(1000)
                records = []
                original = model.forward_with_rgb
                def capture(features, state=None, t=None):
                    output = original(features, state, t)
                    records.append((state.detach().clone(), t.detach().clone(),
                                    {k: v.detach().clone() for k, v in output.items()}))
                    return output
                with patch.object(model, 'encode_rgb', wraps=model.encode_rgb) as rgb, \
                        patch.object(model, 'forward_with_rgb', side_effect=capture):
                    cached = sample_scores(model, diffusion, image, noise, steps=5)
                    self.assertEqual(rgb.call_count, 1)
                    cached_records = records[:]
                    records.clear(); rgb.reset_mock()
                    uncached = diffusion.sample(model, image, noise, steps=5)
                    self.assertEqual(rgb.call_count, 5)
                self.assertTrue(torch.equal(noise, input_noise))
                self.assertTrue(torch.equal(cached, uncached))
                self.assert_tree_equal(cached_records, records)
                self.assertTrue(torch.equal(cached, cached_records[-1][2]['logits']))
                self.assertGreater(float(cached.abs().max()), 1.)
                self.assertFalse(torch.equal(cached, cached_records[-1][2]['x0']))
                self.assertFalse(cached.requires_grad)
                self.assertEqual([int(row[1][0]) for row in records], [999, 749, 500, 250, 0])
                for now, later in zip(cached_records, cached_records[1:]):
                    xt, t, prediction = now
                    a, ap = diffusion.abar[t[0]], diffusion.abar[later[1][0]]
                    x0 = prediction['x0'].clamp(-1, 1)
                    epsilon = (xt-a.sqrt()*x0)/(1-a).sqrt()
                    self.assertTrue(torch.equal(ap.sqrt()*x0+(1-ap).sqrt()*epsilon, later[0]))

    def test_teacher_uses_bounded_x0_independent_noise_and_half_membership(self):
        for mapping in ['tanh', 'simplex']:
            with self.subTest(mapping=mapping):
                seed_cpu(123)
                diffusion = GaussianDiffusion(1000)
                model = DictTeacher(mapping)
                image = torch.randn(4, 3, 8, 8)
                target = torch.randint(4, (4, 8, 8))
                encoded = diffusion.encode(target)
                original_target = encoded.clone()
                t = torch.tensor([100, 300, 910, 980])
                regular_noise = torch.randn_like(encoded)
                noisy, selected = mixed_recycling_input(model, diffusion, image, encoded, t, regular_noise)
                self.assertEqual(selected.unique().numel(), 2)
                teacher = model.calls[-1]
                self.assertFalse(teacher['grad_enabled'])
                self.assertTrue((teacher['t'] == 999).all())
                self.assertFalse(torch.equal(teacher['noise'], regular_noise[selected]))
                self.assertFalse(torch.equal(teacher['noise'], encoded[selected]))
                self.assertFalse(torch.equal(teacher['x0'], teacher['logits'].clamp(-1, 1)))
                expected_source = encoded.clone()
                expected_source[selected] = teacher['x0']
                self.assertTrue(torch.equal(noisy, diffusion.q_sample(expected_source, t, regular_noise)))
                self.assertTrue(torch.equal(encoded, original_target))
                self.assertFalse(noisy.requires_grad)
                self.assertIsNone(model.logits.grad)

    def test_ce_dice_mse_gradients_use_raw_logits_and_bounded_state(self):
        seed_cpu(42)
        image = torch.randn(2, 3, 32, 32)
        state = torch.randn(2, 4, 32, 32)
        time = torch.tensor([139, 971])
        target = torch.randint(4, (2, 32, 32))
        encoded = GaussianDiffusion.encode(target)
        classified_gradients = {}
        for mapping in ['tanh', 'simplex']:
            model = StateObjectiveModel(mapping)
            output = model(image, state, time)
            total, terms = losses(output, target, {'boundary': False}, encoded)
            self.assertTrue(torch.equal(terms['loss_x0_mse'], (output['x0']-encoded).square().mean()))
            self.assertTrue(torch.equal(total, .5*terms['loss_ce']+.5*terms['loss_dice_fg']+terms['loss_x0_mse']))
            named = list(model.named_parameters())
            for term in ['loss_ce', 'loss_dice_fg', 'loss_x0_mse']:
                grads = torch.autograd.grad(terms[term], [p for _, p in named], retain_graph=True)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads), mapping + '/' + term)
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0, mapping + '/' + term)
                for prefix in ['rgb_encoder.', 'decoder.', 'head.', 'state_encoder.', 'state_projection.', 'time_mlp.']:
                    self.assertGreater(sum(float(g.abs().sum()) for (name, _), g in zip(named, grads) if name.startswith(prefix)), 0, prefix)
                if term != 'loss_x0_mse':
                    classified_gradients[mapping, term] = [g.detach().clone() for g in grads]
            total.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        for term in ['loss_ce', 'loss_dice_fg']:
            self.assert_tree_equal(classified_gradients['tanh', term], classified_gradients['simplex', term])

    def test_explicit_binding_is_process_local_and_restored_after_patch(self):
        original_build, original_sampler = shared.build, shared.sample_cached
        with patch.object(shared, 'build', original_build), patch.object(shared, 'sample_cached', original_sampler):
            bind_experiment()
            for mapping in ['tanh', 'simplex']:
                model = shared.build(config(mapping))
                self.assertIsInstance(model, StateObjectiveModel)
                self.assertEqual(model.state_mapping, mapping)
            self.assertIs(shared.sample_cached, sample_scores)
        self.assertIs(shared.build, original_build)
        self.assertIs(shared.sample_cached, original_sampler)

    def test_bound_models_actual_shared_training_step_adam_scheduler_rng_resume(self):
        with patch('torch.cuda.is_available', return_value=False):
            for mapping in ['tanh', 'simplex']:
                with self.subTest(mapping=mapping):
                    c = config(mapping)
                    def update(model, optimizer, scheduler):
                        image = torch.randn(2, 3, 32, 32) * (.75 + random.random()/2)
                        target = torch.from_numpy(np.random.randint(0, 4, (2, 32, 32))).long()
                        terms, _, counts = shared.train_step(model, GaussianDiffusion(1000), optimizer, image, target, c)
                        self.assertEqual(counts['high_noise_count'], 1)
                        self.assertEqual(counts['recycled_count'], 1)
                        scheduler.step()
                        return terms
                    original_build, original_sampler = shared.build, shared.sample_cached
                    with patch.object(shared, 'build', original_build), patch.object(shared, 'sample_cached', original_sampler):
                        bind_experiment()
                        seed_cpu(42)
                        model = shared.build(c)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=1e-4)
                        scheduler = shared.scheduler_for(optimizer, c)
                        update(model, optimizer, scheduler)
                        buffer = io.BytesIO()
                        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                    'scheduler': scheduler.state_dict(), 'rng': rng_state()}, buffer)
                        expected_losses = [update(model, optimizer, scheduler) for _ in range(2)]
                        expected_model = copy.deepcopy(model.state_dict())
                        expected_optimizer = copy.deepcopy(optimizer.state_dict())
                        expected_scheduler = copy.deepcopy(scheduler.state_dict())
                        expected_rng = rng_state()
                        del model, optimizer, scheduler
                        buffer.seek(0)
                        restored = torch.load(buffer, map_location='cpu', weights_only=False)
                        model = shared.build(c)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=1e-4)
                        scheduler = shared.scheduler_for(optimizer, c)
                        model.load_state_dict(restored['model']); optimizer.load_state_dict(restored['optimizer'])
                        scheduler.load_state_dict(restored['scheduler']); restore_rng(restored['rng'])
                        observed_losses = [update(model, optimizer, scheduler) for _ in range(2)]
                        self.assert_tree_equal(expected_losses, observed_losses)
                        self.assert_tree_equal(expected_model, model.state_dict())
                        self.assert_tree_equal(expected_optimizer, optimizer.state_dict())
                        self.assert_tree_equal(expected_scheduler, scheduler.state_dict())
                        self.assert_tree_equal(expected_rng, rng_state())
                        del model, optimizer, scheduler
                    self.assertIs(shared.build, original_build)
                    self.assertIs(shared.sample_cached, original_sampler)

    def test_invalid_mapping_and_sampler_registration_are_rejected(self):
        with self.assertRaises(ValueError):
            StateObjectiveModel('unregistered')
        with self.assertRaises(ValueError):
            bounded_state(torch.zeros(1, 4, 4, 4), 'unregistered')
        with self.assertRaises(TypeError):
            sample_scores(StaticSemanticModel('S2'), GaussianDiffusion(1000),
                          torch.zeros(1, 3, 8, 8), torch.zeros(1, 4, 8, 8))
        with self.assertRaises(ValueError):
            sample_scores(StateObjectiveModel('tanh'), GaussianDiffusion(1000),
                          torch.zeros(1, 3, 8, 8), torch.zeros(1, 4, 8, 8), steps=0)


if __name__ == '__main__':
    unittest.main()

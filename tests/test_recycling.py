import copy
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.recycling import mixed_recycling_input
from conditional_diffusion.trainer import validate_config


class RecordingTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.25))
        self.calls = []

    def forward(self, image, state, t):
        self.calls.append((state.detach().clone(), t.clone(), torch.is_grad_enabled()))
        return self.value.expand_as(state)


class RecyclingTests(unittest.TestCase):
    def setUp(self):
        self.diffusion = GaussianDiffusion(1000)
        self.model = RecordingTeacher()
        self.image = torch.zeros(2, 3, 4, 4)
        self.x0 = self.diffusion.encode(torch.zeros(2, 4, 4, dtype=torch.long))
        self.t = torch.tensor([123, 987])
        self.noise = torch.zeros_like(self.x0)

    def make_input(self):
        return mixed_recycling_input(self.model, self.diffusion, self.image,
                                     self.x0, self.t, self.noise)

    def test_teacher_is_pure_noise_terminal_detached_and_gt_stays_unchanged(self):
        original = self.x0.clone()
        torch.manual_seed(42)
        noisy, selected = self.make_input()
        self.assertEqual(len(selected), 1)
        state, t, grad_enabled = self.model.calls[0]
        self.assertFalse(grad_enabled)
        self.assertTrue(bool((t == 999).all()))
        self.assertGreater(float(state.std()), 0.1)
        self.assertFalse(torch.equal(state, self.x0[selected]))
        self.assertFalse(noisy.requires_grad)
        self.assertTrue(torch.equal(self.x0, original))
        source = original.clone()
        source[selected] = 0.25
        self.assertTrue(torch.equal(noisy, self.diffusion.q_sample(source, self.t, self.noise)))
        loss = self.model(self.image, noisy, self.t).square().mean()
        loss.backward()
        self.assertGreater(float(self.model.value.grad.abs()), 0)
        self.assertTrue(self.model.calls[-1][2])

    def test_teacher_codes_are_bounded(self):
        with torch.no_grad():
            self.model.value.fill_(8)
        noisy, selected = self.make_input()
        expected = self.x0.clone()
        expected[selected] = 1
        self.assertTrue(torch.equal(noisy, self.diffusion.q_sample(expected, self.t, self.noise)))

    def test_membership_varies_independently_of_fixed_time_strata(self):
        torch.manual_seed(42)
        positions = [int(self.make_input()[1][0]) for _ in range(100)]
        self.assertEqual(set(positions), {0, 1})
        self.assertTrue(25 < sum(positions) < 75)

    def test_rng_restore_reproduces_teacher_noise_and_membership(self):
        torch.manual_seed(42)
        rng = torch.get_rng_state()
        first, selection1 = self.make_input()
        teacher_noise1 = self.model.calls[-1][0]
        final_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        second, selection2 = self.make_input()
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(selection1, selection2))
        self.assertTrue(torch.equal(teacher_noise1, self.model.calls[-1][0]))
        self.assertTrue(torch.equal(final_rng, torch.get_rng_state()))

    def test_config_is_one_training_state_factor(self):
        base = json.loads((ROOT / 'configs/pilot_high_noise.json').read_text())
        config = json.loads((ROOT / 'configs/pilot_recycling.json').read_text())
        validate_config(config)
        changed = {k for k in set(base) | set(config) if base.get(k) != config.get(k)}
        self.assertEqual(changed, {'experiment_variant', 'training_state', 'recycle_fraction'})
        self.assertEqual(config['epochs'] * 120 // config['batch_size'], 2400)

    def test_invalid_or_unregistered_recycling_is_rejected(self):
        config = json.loads((ROOT / 'configs/pilot_recycling.json').read_text())
        patches = [{'experiment_variant': 'high_noise_coverage'}, {'training_state': 'ground_truth'},
                   {'training_state': 'unknown'}, {'recycle_fraction': 1.0}, {'batch_size': 3},
                   {'phase': 'P2'}, {'debug': True}, {'method': 'direct'}, {'timestep_sampling': 'uniform'}]
        for patch in patches:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_config(dict(copy.deepcopy(config), **patch))


if __name__ == '__main__':
    unittest.main()

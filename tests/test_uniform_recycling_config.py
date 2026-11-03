import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.trainer import validate_config
sys.path.insert(0, str(ROOT / 'work'))
from launch_p1_uniform_recycling import factorial_effects


class UniformRecyclingConfigTests(unittest.TestCase):
    def test_only_timestep_policy_changes_from_high_noise_recycling(self):
        before = json.loads((ROOT / 'configs/pilot_recycling.json').read_text())
        config = json.loads((ROOT / 'configs/pilot_uniform_recycling.json').read_text())
        validate_config(config)
        changed = {k for k in set(before) | set(config) if before.get(k) != config.get(k)}
        self.assertEqual(changed, {'timestep_sampling', 'experiment_variant'})
        self.assertEqual(config['timestep_sampling'], 'uniform')
        self.assertEqual(config['recycle_fraction'], 0.5)
        self.assertEqual(config['epochs'] * 120 // config['batch_size'], 2400)

    def test_uniform_recycling_keeps_original_pilot_recipe(self):
        base = json.loads((ROOT / 'configs/pilot.json').read_text())
        config = json.loads((ROOT / 'configs/pilot_uniform_recycling.json').read_text())
        self.assertTrue(all(config[k] == v for k, v in base.items()))
        self.assertEqual(set(config) - set(base),
                         {'timestep_sampling', 'high_noise_start', 'experiment_variant', 'training_state', 'recycle_fraction'})

    def test_wrong_policy_or_registration_is_rejected(self):
        base = json.loads((ROOT / 'configs/pilot_uniform_recycling.json').read_text())
        patches = [{'timestep_sampling': 'high_noise_stratified'}, {'experiment_variant': 'high_noise_recycling'},
                   {'experiment_variant': 'unknown'}, {'training_state': 'ground_truth'}, {'recycle_fraction': 0},
                   {'recycle_fraction': 1.0}, {'batch_size': 1}, {'batch_size': 3}, {'high_noise_start': 1000},
                   {'method': 'direct'}, {'phase': 'P2'}, {'debug': True}]
        for patch in patches:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_config(dict(base, **patch))

    def test_factorial_effect_signs_and_interaction(self):
        matrix = {cell: {'miou': value} for cell, value in {
            'uniform_standard': 0.60, 'high_noise_standard': 0.70,
            'uniform_recycling': 0.65, 'high_noise_recycling': 0.80}.items()}
        effects = factorial_effects(matrix, ['miou'])['miou']
        self.assertAlmostEqual(effects['high_noise_without_recycling'], 0.10)
        self.assertAlmostEqual(effects['high_noise_with_recycling'], 0.15)
        self.assertAlmostEqual(effects['recycling_without_high_noise'], 0.05)
        self.assertAlmostEqual(effects['recycling_with_high_noise'], 0.10)
        self.assertAlmostEqual(effects['high_noise_average'], 0.125)
        self.assertAlmostEqual(effects['recycling_average'], 0.075)
        self.assertAlmostEqual(effects['interaction'], 0.05)


if __name__ == '__main__':
    unittest.main()

import unittest,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.diffusion import GaussianDiffusion

class TimestepSamplingTests(unittest.TestCase):
    def test_uniform_preserves_original_draw_and_rng(self):
        d=GaussianDiffusion(1000)
        torch.manual_seed(42)
        expected=torch.randint(1000,(32,));expected_state=torch.get_rng_state()
        torch.manual_seed(42)
        actual=d.training_timesteps(32)
        self.assertTrue(torch.equal(expected,actual))
        self.assertTrue(torch.equal(expected_state,torch.get_rng_state()))

    def test_high_noise_coverage_and_rng_resume(self):
        d=GaussianDiffusion(1000)
        torch.manual_seed(42)
        first=d.training_timesteps(2,'high_noise_stratified',900)
        state=torch.get_rng_state()
        expected=d.training_timesteps(2,'high_noise_stratified',900)
        torch.set_rng_state(state)
        actual=d.training_timesteps(2,'high_noise_stratified',900)
        self.assertTrue(torch.equal(expected,actual))
        draws=torch.cat([first,actual]+[d.training_timesteps(2,'high_noise_stratified',900) for _ in range(98)])
        self.assertEqual(int((draws>=900).sum()),100)
        self.assertTrue(bool(((draws>=0)&(draws<1000)).all()))

    def test_reject_invalid_strata(self):
        d=GaussianDiffusion(1000)
        for batch,policy,cutoff in [(1,'high_noise_stratified',900),(3,'high_noise_stratified',900),(2,'bad',900),(2,'high_noise_stratified',1000)]:
            with self.assertRaises(ValueError):d.training_timesteps(batch,policy,cutoff)

if __name__=='__main__':unittest.main()

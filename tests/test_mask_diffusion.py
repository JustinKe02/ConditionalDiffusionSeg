"""Regression checks for the new mask-space diffusion contract."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from conditional_diffusion.mask_diffusion import GaussianMaskDiffusion, MaskStateCodec


class MaskDiffusionTests(unittest.TestCase):
    def test_ordinal_codes_and_distances(self):
        codec = MaskStateCodec()
        mask = torch.arange(4).reshape(1, 1, 4)
        codes = codec.encode(mask)[0, :, 0].T
        expected = torch.tensor([[-1., -1., -1.], [1., -1., -1.], [1., 1., -1.], [1., 1., 1.]])
        self.assertTrue(torch.equal(codes, expected))
        distances = (codes[:, None] - codes[None]).square().sum(-1)
        ranks = torch.arange(4)
        self.assertTrue(torch.equal(distances, 4 * (ranks[:, None] - ranks[None]).abs()))

    def test_codec_mapping_and_monotonicity(self):
        mask = torch.tensor([[[0, 1], [2, 3]]])
        truth = torch.nn.functional.one_hot(mask, 4).permute(0, 3, 1, 2).float()
        for representation, order in [('onehot', None), ('ordinal', None), ('ordinal', [0, 2, 1, 3])]:
            codec = MaskStateCodec(representation, class_order=order)
            self.assertTrue(torch.equal(codec.encode(mask), codec.from_probabilities(truth)))
        probabilities = torch.randn(2, 4, 5, 7).softmax(1)
        cumulative = MaskStateCodec().from_probabilities(probabilities)
        self.assertTrue(bool((cumulative[:, :-1] >= cumulative[:, 1:]).all()))
        permuted = MaskStateCodec(class_order=[0, 2, 1, 3]).encode(mask)
        self.assertTrue(torch.equal(permuted[0, :, 0, 1], torch.tensor([1., 1., -1.])))

    def test_forward_and_ddim_formula(self):
        diffusion = GaussianMaskDiffusion(MaskStateCodec(), timesteps=1000)
        clean = torch.randn(2, 3, 5, 7)
        noise = torch.randn_like(clean)
        times = torch.tensor([0, 999])
        alpha = diffusion.abar[times, None, None, None]
        self.assertTrue(torch.equal(diffusion.q_sample(clean, times, noise), alpha.sqrt() * clean + (1-alpha).sqrt() * noise))
        self.assertLess(float(diffusion.abar[-1]), 1e-7)
        time, next_time = 700, 200
        state = diffusion.q_sample(clean, torch.full((2,), time), noise)
        expected = diffusion.q_sample(clean, torch.full((2,), next_time), noise)
        self.assertTrue(torch.allclose(diffusion.ddim_step(state, clean, time, next_time), expected, atol=1e-6))

    def test_sampler_returns_semantics_and_uses_matching_states(self):
        for representation in ['ordinal', 'onehot']:
            diffusion = GaussianMaskDiffusion(MaskStateCodec(representation), timesteps=10)
            logits = torch.tensor([2., 5., -3., -4.]).view(1, 4, 1, 1).expand(1, 4, 4, 4)

            class Oracle:
                def __init__(self):
                    self.calls = []

                def denoise(self, condition, state, times):
                    self.calls.append((state.clone(), int(times[0])))
                    return logits

            for steps in [1, 2, 10]:
                model = Oracle()
                noise = torch.randn(1, diffusion.codec.channels, 4, 4)
                original = noise.clone()
                result = diffusion.sample(model, {}, noise, steps)
                self.assertTrue(torch.equal(result, logits.softmax(1)))
                self.assertGreater(float(result[:, 1].min()), .9)
                self.assertTrue(torch.equal(noise, original))
                self.assertEqual(len(model.calls), steps)
                self.assertEqual(model.calls[0][1], 9)
                clean = diffusion.codec.from_probabilities(result)
                for (state, time), (next_state, next_time) in zip(model.calls, model.calls[1:]):
                    self.assertTrue(torch.equal(diffusion.ddim_step(state, clean, time, next_time), next_state))

    def test_loss_backpropagates_through_ordinal_mapping(self):
        diffusion = GaussianMaskDiffusion(MaskStateCodec())
        logits = torch.randn(2, 4, 4, 4, requires_grad=True)
        mask = torch.randint(4, (2, 4, 4))
        terms = diffusion.loss(logits, mask)
        terms['loss_total'].backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA regression requires a GPU')
    def test_cuda_loss_supports_strict_determinism(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            diffusion = GaussianMaskDiffusion(MaskStateCodec(), device='cuda')
            logits = torch.randn(2, 4, 16, 16, device='cuda', requires_grad=True)
            mask = torch.randint(4, (2, 16, 16), device='cuda')
            diffusion.loss(logits, mask)['loss_total'].backward()
            self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_validation(self):
        with self.assertRaises(ValueError):
            MaskStateCodec(class_order=[0, 1, 1, 3])
        for invalid_order in [[False, 1, 2, 3], [0., 1, 2, 3]]:
            with self.assertRaises(ValueError):
                MaskStateCodec(class_order=invalid_order)
        with self.assertRaises(ValueError):
            MaskStateCodec('onehot', class_order=[0, 2, 1, 3])
        with self.assertRaises(ValueError):
            GaussianMaskDiffusion(MaskStateCodec(), timesteps=1)
        diffusion = GaussianMaskDiffusion(MaskStateCodec())
        noise = torch.randn(1, 3, 4, 4)
        for steps in [0, 1001, 2.5, True]:
            with self.assertRaises(ValueError):
                diffusion.sample(None, None, noise, steps)
        with self.assertRaises(ValueError):
            diffusion.q_sample(noise, torch.tensor([1000]), noise)
        with self.assertRaises(ValueError):
            diffusion.q_sample(noise, torch.tensor([1.]), noise)

        class NonfiniteDenoiser:
            def denoise(self, condition, state, times):
                return torch.full((1, 4, 4, 4), float('nan'))

        with self.assertRaises(FloatingPointError):
            diffusion.sample(NonfiniteDenoiser(), None, noise, steps=1)


if __name__ == '__main__':
    unittest.main()

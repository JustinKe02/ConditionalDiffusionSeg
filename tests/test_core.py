import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.inference import positions, predict_image
from conditional_diffusion.metrics_impl import confusion, summarize, boundary_counts, summarize_boundaries
from conditional_diffusion.model import SegUNet
from conditional_diffusion.objectives import losses


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_encoding_and_q_sample_endpoints(self):
        d = GaussianDiffusion(1000)
        mask = torch.arange(4).reshape(1, 2, 2)
        x0 = d.encode(mask)
        self.assertTrue(torch.equal(x0.argmax(1), mask))
        self.assertTrue(torch.all(d.abar[:-1] > d.abar[1:]))
        self.assertLess(float(d.abar[-1]), 1e-6)
        t = torch.tensor([999])
        noise = torch.ones_like(x0)
        self.assertTrue(torch.allclose(d.q_sample(x0, t, noise), d.abar[t].sqrt()*x0+(1-d.abar[t]).sqrt()*noise))

    def test_ddim_oracle_trajectory(self):
        d = GaussianDiffusion(16)
        x0 = d.encode(torch.arange(64).reshape(1, 8, 8)%4)
        noise = torch.randn_like(x0)
        epsilon = (noise-d.abar[-1].sqrt()*x0)/(1-d.abar[-1]).sqrt()
        calls = []
        def oracle(image, xt, t):
            a = d.abar[t][:, None, None, None]
            self.assertTrue(torch.allclose(xt, a.sqrt()*x0+(1-a).sqrt()*epsilon, atol=2e-5))
            calls.append(int(t[0]))
            return x0
        result = d.sample(oracle, torch.zeros(1, 3, 8, 8), noise, 5)
        self.assertTrue(torch.equal(result, x0))
        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[0], 15)
        self.assertEqual(calls[-1], 0)

    def test_coverage_and_padding(self):
        coverage = np.zeros(1922, dtype=int)
        for y in positions(1922, 512, 384):
            coverage[y:y+512] += 1
        self.assertTrue((coverage > 0).all())
        class Constant(torch.nn.Module):
            method = 'direct'
            iterations = 1
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
            def forward(self, image):
                result = image.new_zeros((1, 4, *image.shape[-2:]))
                result[:, 2] = 10
                return result
        prediction, nfe = predict_image(Constant(), None, torch.zeros(3, 19, 37), 'a',
                                       {'crop_size': 32, 'stride': 24, 'samples_per_image': 1})
        self.assertEqual(prediction.shape, (19, 37))
        self.assertTrue((prediction == 2).all())
        self.assertEqual(nfe, 2)

    def test_metrics_and_boundaries(self):
        mask = (np.arange(64).reshape(8, 8)%4).astype(np.uint8)
        matrix = confusion(mask, mask)
        self.assertEqual(summarize(matrix)['miou_4class'], 1.)
        self.assertEqual(summarize_boundaries(boundary_counts(mask, mask))['boundary_f1_foreground'], 1.)

    def test_all_methods_gradients(self):
        for method in ['direct', 'iterative', 'diffusion']:
            model = SegUNet(method, base=8, iterations=2)
            image, mask = torch.randn(2, 3, 32, 32), torch.randint(4, (2, 32, 32))
            d = GaussianDiffusion(16)
            x0 = d.encode(mask) if method == 'diffusion' else None
            logits = model(image, d.q_sample(x0, torch.tensor([0, 15]), torch.randn_like(x0)), torch.tensor([0, 15])) if x0 is not None else model(image)
            loss, _ = losses(logits, mask, {'boundary': True, 'boundary_weight': .1}, x0)
            loss.backward()
            self.assertEqual(logits.shape, (2, 4, 32, 32))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()

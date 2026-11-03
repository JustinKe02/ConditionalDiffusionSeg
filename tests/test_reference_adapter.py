import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.condition_model import ReferenceFusionUNet
from conditional_diffusion.reference_adapter import FrozenReferenceAdapter
from conditional_diffusion.reference_data import reference_patch
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.model import make_model
from conditional_diffusion.trainer import validate_config


class ReferenceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'base.pt'
        c = json.loads((ROOT / 'configs/pilot_reference_f0.json').read_text())
        c['base_channels'] = 8
        self.base = ReferenceFusionUNet(8).eval()
        torch.save({'model': self.base.state_dict(), 'config': c}, self.path)
        self.config = json.loads((ROOT / 'configs/pilot_reference_adapter_b.json').read_text())
        self.config.update(base_channels=8, adapter_width=8, adapter_base_checkpoint=str(self.path),
                           adapter_base_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())

    def tearDown(self):
        self.temp.cleanup()

    def inputs(self, accepted=True):
        image = torch.randn(2, 7, 32, 32)
        image[:, 6] = .8 if accepted else 0
        return image, torch.randn(2, 4, 32, 32), torch.tensor([100, 950])

    def test_zero_initialization_exact_forward_and_ddim(self):
        model = make_model(self.config).eval()
        image, xt, t = self.inputs()
        rgb_only = torch.cat([image[:, :3], torch.zeros_like(image[:, 3:])], dim=1)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(image, xt, t), self.base(rgb_only, xt, t)))
            d = GaussianDiffusion(1000)
            self.assertTrue(torch.equal(d.sample(model, image, xt, 4), d.sample(self.base, rgb_only, xt, 4)))

    def test_trained_adapter_keeps_base_and_rejected_output_exact(self):
        model = make_model(self.config).train()
        old = {k: v.clone() for k, v in model.base.state_dict().items()}
        self.assertFalse(model.base.training)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        image, xt, t = self.inputs()
        image[0, 6] = 0
        rgb_only = torch.cat([image[:, :3], torch.zeros_like(image[:, 3:])], dim=1)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            out = model(image, xt, t)
            out.square().mean().backward()
            self.assertTrue(all(p.grad is None for p in model.base.parameters()))
            opt.step()
        with torch.no_grad():
            after, baseline = model(image, xt, t), self.base(rgb_only, xt, t)
        self.assertTrue(torch.equal(after[0], baseline[0]))
        self.assertFalse(torch.equal(after[1], baseline[1]))
        self.assertTrue(all(torch.equal(v, model.base.state_dict()[k]) for k, v in old.items()))
        image[:, 6] = 0
        self.assertFalse(model(image, xt, t).requires_grad)
        d = GaussianDiffusion(1000)
        with torch.no_grad():
            self.assertTrue(torch.equal(d.sample(model, image, xt, 4), d.sample(self.base, rgb_only, xt, 4)))

    def test_matched_initialization_rng_and_content_ablation(self):
        torch.manual_seed(42)
        a = make_model(dict(self.config, adapter_reference_enabled=False))
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        b = make_model(self.config)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
        with torch.no_grad():
            a.adapter.output.weight.fill_(.1)
            b.load_state_dict(a.state_dict())
        image, xt, t = self.inputs()
        changed = image.clone(); changed[:, 3:6] += 1
        with torch.no_grad():
            self.assertTrue(torch.equal(a(image, xt, t), a(changed, xt, t)))
            self.assertFalse(torch.equal(b(image, xt, t), b(changed, xt, t)))

    def test_reference_weight_applied_once_and_rejection_zeroed(self):
        entry = dict(size_wh=[16, 16], accepted=True, rgb_confidence=.8,
                     coefficient_rgb=[[.5]*3] + [[0]*3]*5)
        rgb = np.full((16, 16, 3), 128, np.uint8)
        weighted = reference_patch(rgb, entry)
        raw = reference_patch(rgb, entry, contrast_weighted=False)
        self.assertTrue(np.allclose(weighted[..., :3], raw[..., :3] * .8))
        self.assertTrue(np.array_equal(weighted[..., 3], raw[..., 3]))
        entry['accepted'] = False
        self.assertEqual(int(np.count_nonzero(reference_patch(rgb, entry, contrast_weighted=False))), 0)

    def test_nonconstant_confidence_and_changed_checkpoint_rejected(self):
        model = make_model(self.config)
        image, xt, t = self.inputs()
        image[0, 6, 1, 1] = 0
        with self.assertRaises(ValueError):
            model(image, xt, t)
        self.path.write_bytes(self.path.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            make_model(self.config)

    def test_config_rejects_unregistered_or_unsafe_combinations(self):
        c = json.loads((ROOT / 'configs/pilot_reference_adapter_b.json').read_text())
        validate_config(c)
        for patch in [dict(adapter_reference_enabled=False), dict(reference_contrast_weighted=True),
                      dict(reference_enabled=False), dict(condition_fusion='adaptive'),
                      dict(condition_architecture='split_rgb_state'), dict(debug=True), dict(dual_head=True),
                      dict(training_state='ground_truth'), dict(adapter_base_sha256='')]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_config(dict(c, **patch))


if __name__ == '__main__':
    unittest.main()

import json
import sys
import unittest
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.model import make_model
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.objectives import losses
from conditional_diffusion.recycling import mixed_recycling_input
from conditional_diffusion.trainer import validate_config


class Probe(torch.nn.Module):
    def __init__(self, semantic=9.):
        super().__init__()
        self.semantic = semantic
        self.states = []

    def forward(self, image, state, t):
        self.states.append((state.clone(), t.clone(), torch.is_grad_enabled()))
        return {'logits': torch.full_like(state, self.semantic), 'x0': torch.full_like(state, .25)}


class DualHeadTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / 'configs/pilot_dual_head.json').read_text())

    def test_registered_config_only_changes_heads(self):
        c = self.config(); validate_config(c)
        base = json.loads((ROOT / 'configs/pilot_recycling.json').read_text())
        self.assertEqual({k for k in set(c)|set(base) if c.get(k) != base.get(k)}, {'dual_head', 'experiment_variant'})

    def test_reject_wrong_registration(self):
        for patch in [{'dual_head': False}, {'dual_head': 1}, {'experiment_variant': 'high_noise_recycling'},
                      {'timestep_sampling': 'uniform'}, {'training_state': 'ground_truth'},
                      {'method': 'direct'}, {'phase': 'P2'}, {'debug': True}]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_config(dict(self.config(), **patch))

    def test_initialization_is_matched_without_rng_change(self):
        c = self.config(); c['base_channels'] = 8
        torch.manual_seed(42); old = make_model(dict(c, dual_head=False)); rng = torch.get_rng_state()
        torch.manual_seed(42); new = make_model(c)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items()))
        self.assertTrue(all(torch.equal(a, b) and a.data_ptr() != b.data_ptr()
                            for a, b in zip(new.out.parameters(), new.denoise_out.parameters())))
        self.assertEqual(sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in old.parameters()), 52)
        image, xt, t = torch.randn(2,3,32,32), torch.randn(2,4,32,32), torch.tensor([1,999])
        with torch.no_grad():
            a, b = old(image,xt,t), new(image,xt,t)
        self.assertTrue(torch.equal(a,b['logits']) and torch.equal(a,b['x0']))

    def test_loss_gradients_are_separated(self):
        semantic = torch.randn(2,4,8,8,requires_grad=True)
        denoise = torch.randn(2,4,8,8,requires_grad=True)
        target = torch.randint(4,(2,8,8)); gt = GaussianDiffusion.encode(target)
        total, terms = losses({'logits':semantic,'x0':denoise},target,self.config(),gt)
        gradients = torch.autograd.grad(terms['loss_ce']+terms['loss_dice_fg'],[semantic,denoise],retain_graph=True,allow_unused=True)
        self.assertIsNotNone(gradients[0]); self.assertIsNone(gradients[1])
        gradients = torch.autograd.grad(terms['loss_x0_mse'],[semantic,denoise],retain_graph=True,allow_unused=True)
        self.assertIsNone(gradients[0]); self.assertIsNotNone(gradients[1])
        self.assertTrue(torch.equal(terms['loss_x0_mse'], torch.nn.functional.mse_loss(denoise,gt)))
        total.backward(); self.assertTrue(torch.isfinite(semantic.grad).all() and torch.isfinite(denoise.grad).all())

    def test_shared_backbone_and_both_heads_train(self):
        c = self.config(); c['base_channels'] = 8; model = make_model(c)
        mask = torch.randint(4,(2,32,32)); image = torch.randn(2,3,32,32)
        output = model(image,torch.randn(2,4,32,32),torch.tensor([100,950]))
        loss,_ = losses(output,mask,c,GaussianDiffusion.encode(mask)); loss.backward()
        for p in [model.input.weight, model.out[-1].weight, model.denoise_out[-1].weight]:
            self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(float(p.grad.abs().sum()),0)

    def test_ddim_uses_x0_states_and_returns_raw_semantic_scores(self):
        d = GaussianDiffusion(10); image = torch.zeros(1,3,8,8); noise = torch.randn(1,4,8,8)
        a,b = Probe(9),Probe(-7)
        self.assertTrue(bool((d.sample(a,image,noise,3)==9).all()))
        self.assertTrue(bool((d.sample(b,image,noise,3)==-7).all()))
        self.assertEqual(len(a.states),3)
        self.assertTrue(all(torch.equal(x[0],y[0]) for x,y in zip(a.states,b.states)))
        self.assertTrue(all(not x[2] for x in a.states))

    def test_recycling_uses_detached_denoising_head(self):
        d=GaussianDiffusion(1000); model=Probe(); image=torch.zeros(2,3,8,8)
        mask=torch.zeros(2,8,8,dtype=torch.long); x0=d.encode(mask); original=x0.clone()
        t=torch.tensor([100,950]); noise=torch.randn_like(x0)
        noisy,selected=mixed_recycling_input(model,d,image,x0,t,noise)
        expected=original.clone(); expected[selected]=.25
        self.assertTrue(torch.equal(noisy,d.q_sample(expected,t,noise)))
        self.assertTrue(torch.equal(x0,original)); self.assertFalse(noisy.requires_grad)
        self.assertTrue(bool((model.states[0][1]==999).all())); self.assertFalse(model.states[0][2])


if __name__ == '__main__': unittest.main()

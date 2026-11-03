"""CPU engineering tests of new probability semantics, warmstart and recovery."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import copy
import io
import json
import random
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from conditional_diffusion.material_categorical import MaterialModel,CategoricalChain,normalize,sample,train_step,categorical
from conditional_diffusion.material_conditions import load_direct,substrate,conditions_for_region
from conditional_diffusion.material_experiment import config_for
from conditional_diffusion.static_experiment import scheduler_for
from conditional_diffusion.runtime import seed_all,rng_state,restore_rng,source_hashes,write_json

PRIOR=R/'outputs/P1/direct/seed_42/p1_cosine_20261017/checkpoints/best.pt'
STATS=dict(mean=[[0,0,0],[.1,.12,.08],[.2,.3,.15],[.4,.1,.2]],
 covariance=[np.eye(3).tolist() for _ in range(4)],similarity=(np.eye(4)+.2).tolist(),
 global_relation=normalize(torch.eye(4)+.2).tolist())


def fixture(side=16):
    image=torch.randn(2,3,side,side);condition=torch.randn(2,8,side,side)*.1
    condition[:,3]=torch.rand(2,side,side);condition[:,4:]=torch.randn(2,4,side,side).softmax(1)
    target=torch.randint(4,(2,side,side));return image,condition,target


class Tests(unittest.TestCase):
    def setUp(self):seed_all(42);torch.set_num_threads(2)

    def test_row_and_terminal_distribution(self):
        chain=CategoricalChain(STATS);_,c,_=fixture(2);r,p=chain.relation(c)
        for t in [1,25,50,99,100]:
            q=chain.transition(r,p,t)
            self.assertTrue(bool((q>=0).all()));torch.testing.assert_close(q.sum(-1),torch.ones_like(p),atol=2e-6,rtol=0)
        torch.testing.assert_close(chain.transition(r,p,100),p[...,None,:].expand_as(r),atol=0,rtol=0)

    def test_variable_time_prefix_and_skip_products(self):
        chain=CategoricalChain(STATS,timesteps=8);_,c,_=fixture(2);r,p=chain.relation(c);t=torch.tensor([1,7])
        prev,total,q=chain.training_matrices(r,p,t)
        for n in range(2):
            a=torch.eye(4).expand(2,2,4,4).clone();b=a.clone()
            for k in range(1,int(t[n])+1):b=a.clone();a=normalize(a@chain.transition(r[n],p[n],k))
            torch.testing.assert_close(prev[n],b,atol=2e-6,rtol=0);torch.testing.assert_close(total[n],a,atol=2e-6,rtol=0)
        prefix,intervals=chain.inference_matrices(r,p,[8,5,2,1])
        for earlier,later in intervals:
            torch.testing.assert_close(normalize(prefix[earlier]@intervals[(earlier,later)]),prefix[later],atol=3e-6,rtol=0)

    def test_exact_bayes_mixture_not_soft_substitution(self):
        chain=CategoricalChain(STATS,timesteps=8);_,c,_=fixture(2);r,p=chain.relation(c)
        prefix,intervals=chain.inference_matrices(r,p,[8,5,2,1]);a=prefix[2];b=intervals[(2,5)];full=prefix[5]
        observed=torch.randint(4,(2,2,2));clean=torch.randn(2,2,2,4).softmax(-1)
        actual=chain.posterior(clean,observed,a,full,b);reference=torch.zeros_like(actual)
        for n in range(2):
            for h in range(2):
                for w in range(2):
                    j=observed[n,h,w]
                    for y in range(4):
                        v=a[n,h,w,y]*b[n,h,w,:,j];v=v/v.sum()
                        reference[n,h,w]+=clean[n,h,w,y]*v
        torch.testing.assert_close(actual,reference,atol=2e-6,rtol=1e-5)

    def test_reliability_zero_equals_global_and_color_changes_local(self):
        chain=CategoricalChain(STATS);_,c,_=fixture(2);c[:,3]=0
        local,p=chain.relation(c,True);global_,_=chain.relation(c,False)
        torch.testing.assert_close(local,global_,atol=0,rtol=0)
        c[:,3]=1;r1,_=chain.relation(c);c[:,:3]+=1.;r2,_=chain.relation(c)
        self.assertGreater(float((r1-r2).abs().max()),.001)

    def test_warmstart_single_forward_and_identical_refiner_initialization(self):
        seed_all(42);before=torch.get_rng_state();teacher=load_direct(PRIOR)
        models={v:MaterialModel(v,str(PRIOR)) for v in ['N0','N1','N2','N3']}
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        image,c,_=fixture();models['N0'].eval()
        with torch.no_grad():torch.testing.assert_close(models['N0'].deterministic(image,c),teacher(image),atol=0,rtol=0)
        for v in ['N2','N3']:
            self.assertEqual(models[v].state_dict().keys(),models['N1'].state_dict().keys())
            for k,x in models[v].state_dict().items():self.assertTrue(torch.equal(x,models['N1'].state_dict()[k]),k)
        self.assertEqual(len(teacher.state_dict())-1,136)

    def test_complete_optimizer_scheduler_rng_restore_all_modes(self):
        for variant in ['N0','N1','N2','N3']:
            seed_all(42);model=MaterialModel(variant);chain=CategoricalChain(STATS);c=config_for(R,variant);c['device']='cpu'
            opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);sched=scheduler_for(opt,c)
            image,condition,target=fixture()
            train_step(model,chain,opt,image,condition,target,c);sched.step()
            # Check loss really reaches every major branch, including new optical path.
            for family in ['rgb_encoder','decoder','head','condition_projection']+([] if variant=='N0' else ['state_encoder','state_projection','time_mlp']):
                self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(family) and p.grad is not None),0.,variant+family)
            payload=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_state()))
            expected=train_step(model,chain,opt,image,condition,target,c);sched.step()
            expected_state=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict());expected_random=(random.random(),np.random.random(),torch.rand(3))
            restored=MaterialModel(variant);restored.load_state_dict(payload['model']);opt2=torch.optim.AdamW(restored.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);sched2=scheduler_for(opt2,c);opt2.load_state_dict(payload['optimizer']);sched2.load_state_dict(payload['scheduler']);restore_rng(payload['rng'])
            actual=train_step(restored,chain,opt2,image,condition,target,c);sched2.step()
            self.assertEqual(expected,actual)
            for k,v in restored.state_dict().items():self.assertTrue(torch.equal(v,expected_state[k]),variant+k)
            for k,v in opt2.state_dict()['state'].items():
                for name,x in v.items():self.assertTrue(torch.equal(x,expected_opt['state'][k][name]),variant+name)
            self.assertEqual(sched2.state_dict(),sched.state_dict());self.assertEqual(random.random(),expected_random[0]);self.assertEqual(np.random.random(),expected_random[1]);self.assertTrue(torch.equal(torch.rand(3),expected_random[2]))

    def test_sampling_private_noise_cache_and_single_rgb_encode(self):
        chain=CategoricalChain(STATS);image,c,_=fixture();uniform=torch.rand(2,5,16,16);saved=uniform.clone()
        for variant in ['N2','N3']:
            model=MaterialModel(variant).eval();calls=[];handle=model.rgb_encoder.input.register_forward_hook(lambda *args:calls.append(1))
            state=torch.get_rng_state();p=sample(model,chain,image,c,uniform);handle.remove()
            self.assertEqual(len(calls),1);self.assertTrue(torch.equal(state,torch.get_rng_state()));self.assertTrue(torch.equal(uniform,saved))
            torch.testing.assert_close(p.sum(1),torch.ones(2,16,16),atol=2e-6,rtol=0)
            q=sample(model,chain,image,c,uniform);self.assertTrue(torch.equal(p,q))

    def test_substrate_no_labels_and_unreliable_fallback(self):
        rgb=np.full((64,64,3),128,np.uint8);prior=np.zeros((4,64,64),np.float32);prior[1]=1
        meta=substrate(rgb,prior);self.assertEqual(meta['reliability'],0.)
        aux=conditions_for_region(rgb,prior,meta,8,16,32,32)
        self.assertEqual(aux.shape,(8,32,32));self.assertEqual(float(aux[3].max()),0.);self.assertTrue(np.isfinite(aux).all())
        np.testing.assert_allclose(aux[4:].sum(0),1.,atol=1e-6)


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    output=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),source_sha256=source_hashes(R),
                scope='new CPU math/warmstart/gradients/full optimizer-scheduler-RNG recovery only; no effect training/validation/test')
    path=R/'outputs/launches/p1_material_categorical_20261020/cpu_verification.json';write_json(path,output)
    print(json.dumps({k:v for k,v in output.items() if k!='source_sha256'}))
    raise SystemExit(0 if result.wasSuccessful() else 1)

"""Replay a real old single-head checkpoint against the source backup."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json,importlib.util
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.model import make_model
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.objectives import losses
from conditional_diffusion.recycling import mixed_recycling_input
from conditional_diffusion.runtime import seed_all,write_json

def old(name):
    spec=importlib.util.spec_from_file_location('legacy_'+name,ROOT/'work/backups/p1_dual_head'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    torch.set_num_threads(4);seed_all(42)
    state=torch.load(ROOT/'outputs/P1/diffusion/seed_42/p1_recycling_20261017/checkpoints/best.pt',map_location='cpu',weights_only=False)
    c=state['config'];a=old('model').make_model(c);b=make_model(c)
    a.load_state_dict(state['model']);b.load_state_dict(state['model']);a.eval();b.eval()
    da=old('diffusion').GaussianDiffusion(1000);db=GaussianDiffusion(1000)
    image=torch.randn(2,3,32,32);noise=torch.randn(2,4,32,32);t=torch.tensor([500,999]);mask=torch.randint(4,(2,32,32));x0=db.encode(mask)
    with torch.no_grad():
        pa,pb=a(image,noise,t),b(image,noise,t);assert torch.equal(pa,pb)
        assert torch.equal(da.sample(a,image,noise,3),db.sample(b,image,noise,3))
    la,ta=old('objectives').losses(pa,mask,c,x0);lb,tb=losses(pb,mask,c,x0)
    assert torch.equal(la,lb) and all(torch.equal(v,tb[k]) for k,v in ta.items())
    torch.manual_seed(123);na,sa=old('recycling').mixed_recycling_input(a,da,image,x0,t,noise);ra=torch.get_rng_state()
    torch.manual_seed(123);nb,sb=mixed_recycling_input(b,db,image,x0,t,noise)
    assert torch.equal(na,nb) and torch.equal(sa,sb) and torch.equal(ra,torch.get_rng_state())
    write_json(ROOT/'outputs/launches/p1_dual_head_20261018/legacy_verification.json',dict(status='passed',
        actual_old_checkpoint_load=True,forward_bitwise_equal=True,ddim_bitwise_equal=True,
        losses_bitwise_equal=True,recycling_and_rng_bitwise_equal=True,scope='Old best weights, CPU32 synthetic inputs; compatibility only, no validation/test performance'))
    print('LEGACY_COMPATIBILITY_PASSED',flush=True)

if __name__=='__main__':main()

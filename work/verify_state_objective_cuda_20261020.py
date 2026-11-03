"""Isolated real512 CUDA checks; weights never used for effect training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'work'))
import torch
from state_objective_worker import make_config
from conditional_diffusion import static_experiment as shared
from conditional_diffusion.state_objective import bind_experiment,build_state_model,sample_scores
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.data import MaterialDataset
from conditional_diffusion.runtime import source_hashes,rng_state,restore_rng,StorageGuard,atomic_checkpoint,write_json


def main():
    bind_experiment();batch=ROOT/'outputs/launches/p1_state_objective_20261020'
    source=source_hashes(ROOT);results={}
    for mapping in ['tanh','simplex']:
        c=make_config(mapping);shared.configure(c)
        directory=batch/f'cuda_check_{mapping}_{time.time_ns()}';directory.mkdir()
        dataset=MaterialDataset(c['manifest'],'train',crop=512,seed=42);dataset.epoch=1
        records=[dataset[i] for i in [0,1]];image=torch.stack([x[0] for x in records]).cuda();target=torch.stack([x[1] for x in records]).cuda()
        model=build_state_model(c).train();diffusion=GaussianDiffusion(1000,'cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        scheduler=shared.scheduler_for(optimizer,c)
        shared.train_step(model,diffusion,optimizer,image,target,c);scheduler.step()
        path=directory/'last.pt';guard=StorageGuard(ROOT,directory,c['storage'])
        atomic_checkpoint(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                          rng=rng_state(),config=c),guard)
        expected_step=shared.train_step(model,diffusion,optimizer,image,target,c)
        expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        expected_adam={i:{k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in d.items()}
                       for i,d in optimizer.state_dict()['state'].items()}
        state=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);restore_rng(state['rng'])
        resumed_step=shared.train_step(model,diffusion,optimizer,image,target,c)
        assert expected_step==resumed_step
        assert all(torch.equal(v.cpu(),expected[k]) for k,v in model.state_dict().items())
        for i,d in optimizer.state_dict()['state'].items():
            for k,v in d.items():
                if torch.is_tensor(v):assert torch.equal(v.cpu(),expected_adam[i][k])
                else:assert v==expected_adam[i][k]
        model.eval();noise=torch.randn(2,4,512,512,device='cuda');before=noise.clone()
        uncached=diffusion.sample(model,image,noise,5);cached=sample_scores(model,diffusion,image,noise,5)
        assert torch.equal(uncached,cached) and torch.equal(noise,before)
        with torch.no_grad():
            output=model(image,noise,torch.tensor([250,999],device='cuda'))
            assert output['x0'].min()>=-1 and output['x0'].max()<=1
            if mapping=='simplex':assert torch.allclose(output['x0'].sum(1),torch.full_like(output['x0'][:,0],-2),atol=1e-6)
        results[mapping]=dict(status='passed',crop=512,batch=2,teacher_included=True,restored_step_weights_adam_bitwise_equal=True,
            cached_uncached_five_steps_raw_readout_bitwise_equal=True,noise_unchanged=True,engineering_optimizer_steps=3,
            parameters=sum(p.numel() for p in model.parameters()),checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        del model,optimizer,scheduler,state,expected,expected_adam,image,target,records,noise,before,uncached,cached,output
        torch.cuda.empty_cache()
    assert source_hashes(ROOT)==source
    write_json(batch/'cuda_verification.json',dict(status='passed',results=results,source_sha256=source,
               effect_training_runs=0,scope='Train-only real512 engineering restore and DDIM checks; all weights isolated'))
    print('Both CUDA state-objective paths passed',flush=True)


if __name__=='__main__':main()

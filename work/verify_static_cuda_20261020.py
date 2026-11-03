"""Isolated real-crop CUDA restore and cached DDIM checks, never an effect run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import sys
import hashlib
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'work'))
import torch
from static_semantic_worker import base_config
from conditional_diffusion.static_experiment import configure, build, train_step, scheduler_for
from conditional_diffusion.static_semantic import sample_cached
from conditional_diffusion.data import MaterialDataset
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.runtime import rng_state,restore_rng,write_json,StorageGuard,atomic_checkpoint,source_hashes


def main():
    batch = ROOT/'outputs/launches/p1_static_semantic_20261020'
    run = batch/'cuda_preflight'
    if run.exists():
        run = batch/('cuda_preflight_'+str(time.time_ns()))
    run.mkdir(exist_ok=False)
    c = base_config('S2',2); configure(c)
    before = source_hashes(ROOT)
    dataset = MaterialDataset(c['manifest'],'train',crop=512,seed=42); dataset.epoch=1
    records = [dataset[i] for i in [0,1]]
    image = torch.stack([x[0] for x in records]).cuda()
    target = torch.stack([x[1] for x in records]).cuda()
    model = build(c).train(); diffusion = GaussianDiffusion(1000,'cuda')
    optimizer = torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scheduler = scheduler_for(optimizer,c)
    train_step(model,diffusion,optimizer,image,target,c); scheduler.step()
    guard = StorageGuard(ROOT,run,c['storage'])
    path = run/'last.pt'
    atomic_checkpoint(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                      rng=rng_state(),config=c),guard)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    uninterrupted = train_step(model,diffusion,optimizer,image,target,c)
    expected = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    expected_optimizer = {i:{k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in x.items()}
                          for i,x in optimizer.state_dict()['state'].items()}
    state = torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'])
    resumed = train_step(model,diffusion,optimizer,image,target,c)
    assert uninterrupted == resumed, 'Restored loss/gradient/count mismatch'
    assert all(torch.equal(v.detach().cpu(),expected[k]) for k,v in model.state_dict().items()), 'CUDA weights differ after restore'
    for i, x in optimizer.state_dict()['state'].items():
        for k,v in x.items():
            assert torch.equal(v.detach().cpu(),expected_optimizer[i][k]) if torch.is_tensor(v) else v == expected_optimizer[i][k]
    model.eval(); noise = torch.randn((2,4,512,512),device='cuda'); initial_noise = noise.clone()
    with torch.no_grad():
        uncached = diffusion.sample(model,image,noise,5)
        cached = sample_cached(model,diffusion,image,noise,5)
    assert torch.equal(uncached,cached), 'CUDA cached/uncached DDIM differ'
    assert torch.equal(noise,initial_noise) and torch.isfinite(cached).all()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest and source_hashes(ROOT) == before
    write_json(batch/'cuda_verification.json',dict(status='passed',device=torch.cuda.get_device_name(),
        image_ids=[x[2] for x in records],crop=512,batch=2,mode='S2',
        engineering_optimizer_steps=3,effect_training_runs=0,teacher_included=True,
        same_input_restored_loss_gradient_counts_bitwise_equal=True,restored_weights_adam_bitwise_equal=True,
        cached_uncached_five_ddim_bitwise_equal=True,max_abs_difference=float((uncached-cached).abs().max()),
        checkpoint_sha256=digest,source_sha256=before,precision='FP32 tensors; same historical backend flags',
        purpose='Train-only engineering check; isolated initialized weights and checkpoint never used by effect runs'))
    print('CUDA restore/cache verification passed',flush=True)


if __name__ == '__main__':
    main()

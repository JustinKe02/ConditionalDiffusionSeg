"""One isolated P0/P1 benchmark, train, or same-best3408 validation process."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from conditional_diffusion.data import MaterialDataset
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.runtime import write_json,source_hashes
from conditional_diffusion import static_experiment as shared
from conditional_diffusion.state_objective import bind_experiment,build_state_model


def make_config(mapping):
    c = json.loads((ROOT/'configs/pilot_static_semantic_s2.json').read_text())
    c.update(experiment_variant='state_objective_'+mapping, state_mapping=mapping,
             architecture='static_semantic_v1_bounded_state', sampler_final_readout='raw_logits',
             state_objective='0.5CE(raw logits)+0.5foregroundDice(softmax logits)+MSE(bounded state,encoded GT)',
             comparison_scope='P0 tanh vs P1 simplex; same raw semantic readout; historical S2 has clipped readout')
    return c


def benchmark(c):
    shared.configure(c)
    model=build_state_model(c).train();diffusion=GaussianDiffusion(1000,'cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    dataset=MaterialDataset(c['manifest'],'train',crop=512,seed=42)
    iterator=iter(shared.loader_for(dataset,c,1));epoch=1
    def step():
        nonlocal iterator,epoch
        try:image,target,_=next(iterator)
        except StopIteration:
            epoch+=1;iterator=iter(shared.loader_for(dataset,c,epoch));image,target,_=next(iterator)
        shared.train_step(model,diffusion,optimizer,image.cuda(),target.cuda(),c)
    for _ in range(10):step()
    shared.synchronize('cuda');torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    for _ in range(30):step()
    shared.synchronize('cuda');elapsed=time.perf_counter()-begin
    result=dict(status='passed',mapping=c['state_mapping'],batch=2,warmup_updates=10,timed_updates=30,
                samples_per_second=60/elapsed,seconds_per_update=elapsed/30,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                total_memory_mib=torch.cuda.get_device_properties(0).total_memory/1024**2,
                scope='engineering only; real IO/crops/augmentation/teacher/loss/backward/clip/AdamW; fresh effect initialization')
    assert result['peak_reserved_mib'] <= .9*result['total_memory_mib']
    return result


def window_benchmark(c):
    shared.configure(c);model=build_state_model(c).eval();diffusion=GaussianDiffusion(1000,'cuda')
    dataset=MaterialDataset(c['manifest'],'val',augment=False)
    begin=time.perf_counter();image,target,ident=dataset[0];read_seconds=time.perf_counter()-begin
    shared.predict(model,diffusion,image[:,:512,:512],ident,c)
    shared.synchronize('cuda');torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    prediction,calls,timing,_=shared.predict(model,diffusion,image,ident,c)
    shared.synchronize('cuda');elapsed=time.perf_counter()-begin
    from conditional_diffusion.metrics_impl import confusion,boundary_counts
    begin=time.perf_counter();confusion(target.numpy(),prediction);boundary_counts(target.numpy(),prediction,2)
    return dict(status='passed',mapping=c['state_mapping'],window_batch_size=1,image_id=ident,
                read_seconds=read_seconds,seconds_without_io_or_metrics=elapsed,metrics_seconds=time.perf_counter()-begin,
                **timing,nfe=calls,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                scope='engineering only; initialized weights; native image; no model selection; labels enter metrics only')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['benchmark','window_benchmark','train','secondary'])
    parser.add_argument('--variant',choices=['P0','P1'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan',type=Path)
    args=parser.parse_args();mapping={'P0':'tanh','P1':'simplex'}[args.variant]
    bind_experiment();c=make_config(mapping)
    try:
        if args.action=='benchmark':result=benchmark(c)
        elif args.action=='window_benchmark':result=window_benchmark(c)
        else:
            plan=json.loads(args.plan.read_text());c=plan['configs'][args.variant];run=Path(plan['runs'][args.variant])
            assert source_hashes(ROOT)==plan['source_sha256'], 'Current source/config differs from effect freeze'
            assert c['state_mapping']==mapping
            if args.action=='train':
                shared.train_static(c,ROOT,run,plan['source_sha256'],resume=run.exists())
                result=dict(status='completed',run=str(run))
            else:
                shared.configure(c);model=build_state_model(c)
                state=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)
                assert state['config']==c and state['source_sha256']==plan['source_sha256']
                model.load_state_dict(state['model']);evaluation=dict(c,sampling_seed=3408)
                metrics,matrices,_,timing=shared.evaluate_static(model,GaussianDiffusion(1000,'cuda'),
                    MaterialDataset(c['manifest'],'val',augment=False),evaluation)
                np.savez_compressed(args.output.with_suffix('.npz'),**matrices)
                write_json(args.output.with_name(args.output.stem+'_timing.json'),timing)
                result=dict(status='completed',epoch=state['epoch'],sampling_seed=3408,metrics=metrics,
                            purpose='same3407-selected best, no reselection or new training seed')
        write_json(args.output,result)
    except BaseException as exc:
        write_json(args.output,dict(status='failed',error=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()

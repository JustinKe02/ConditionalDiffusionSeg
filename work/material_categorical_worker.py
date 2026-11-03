"""Isolated engineering/benchmark/train/secondary worker for the four-arm batch."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion import material_experiment as ex
from conditional_diffusion.material_categorical import MaterialModel,sample,train_step
from conditional_diffusion.material_conditions import ConditionDataset,load_direct,substrate,conditions_for_region,digest
from conditional_diffusion.data import MaterialDataset,MEAN,STD
from conditional_diffusion.runtime import write_json,rng_state,restore_rng,atomic_checkpoint,StorageGuard,source_hashes
from conditional_diffusion.static_experiment import configure,synchronize,scheduler_for,predict as prior_predict

BATCH=ROOT/'outputs/launches/p1_material_categorical_20261020'


def make_config(variant,batch=2,window_batch=1):
    return dict(ex.config_for(ROOT,variant,batch,window_batch),project=str(ROOT))


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def equal_tree(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def verify_cuda():
    rows={};c=make_config('N0');configure(c)
    dataset=ConditionDataset(c['manifest'],c['condition_cache'],'train',512);dataset.epoch=1
    old=MaterialDataset(c['manifest'],'train',512,seed=42);old.epoch=1
    items=[dataset[i] for i in [0,1]]
    for i,item in enumerate(items):
        legacy=old[i];assert torch.equal(item[0],legacy[0]) and torch.equal(item[2],legacy[1]) and item[3]==legacy[2]
        assert item[1].shape==(8,512,512) and torch.isfinite(item[1]).all()
    image,conditions,target=[torch.stack([row[k] for row in items]).cuda() for k in [0,1,2]]
    for variant in ['N0','N1','N2','N3']:
        c=make_config(variant);configure(c);model=ex.build(c);chain=ex.chain_for(c)
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);scheduler=scheduler_for(optimizer,c)
        first=train_step(model,chain,optimizer,image,conditions,target,c);scheduler.step()
        guard=StorageGuard(ROOT,BATCH/('cuda_check_'+variant),c['storage']);guard.run.mkdir(exist_ok=True)
        path=guard.run/'last.pt'
        if path.exists():path.rename(path.with_name('last_previous_'+str(time.time_ns())+'.pt'))
        payload=cpu_tree(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),config=c))
        atomic_checkpoint(path,payload,guard)
        expected=train_step(model,chain,optimizer,image,conditions,target,c);scheduler.step()
        state=cpu_tree(model.state_dict());adam=cpu_tree(optimizer.state_dict());schedule=copy.deepcopy(scheduler.state_dict());random_after=rng_state()
        del model,optimizer,scheduler,payload;torch.cuda.empty_cache()
        saved=torch.load(path,map_location='cpu',weights_only=False)
        restored=ex.build(c);opt2=torch.optim.AdamW(restored.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);sch2=scheduler_for(opt2,c)
        restored.load_state_dict(saved['model']);opt2.load_state_dict(saved['optimizer']);sch2.load_state_dict(saved['scheduler']);restore_rng(saved['rng'])
        actual=train_step(restored,chain,opt2,image,conditions,target,c);sch2.step()
        assert actual==expected,(variant,'loss/gradient restoration mismatch')
        assert equal_tree(cpu_tree(restored.state_dict()),state),(variant,'model restoration mismatch')
        assert equal_tree(cpu_tree(opt2.state_dict()),adam),(variant,'Adam restoration mismatch')
        assert sch2.state_dict()==schedule
        assert torch.equal(rng_state()['torch'],random_after['torch'])
        assert all(torch.equal(a,b) for a,b in zip(rng_state()['cuda'],random_after['cuda']))
        restored.eval();uniform=torch.rand((2,5,512,512),device='cuda',generator=torch.Generator(device='cuda').manual_seed(3407));noise=uniform.clone()
        counts=[];handle=restored.rgb_encoder.input.register_forward_hook(lambda *args:counts.append(1))
        with torch.no_grad():a=sample(restored,chain,image,conditions,uniform)
        handle.remove();assert len(counts)==1 and torch.equal(uniform,noise)
        with torch.no_grad():b=sample(restored,chain,image,conditions,uniform)
        assert torch.equal(a,b) and bool(torch.isfinite(a).all())
        torch.testing.assert_close(a.sum(1),torch.ones_like(a[:,0]),atol=2e-6,rtol=0)
        rows[variant]=dict(status='passed',real_crop=512,batch=2,initial_loss=first[0],restore_exact=True,
                           sampling_repeat_exact=True,rgb_calls=1,checkpoint=str(path),checkpoint_sha256=digest(path),
                           parameters=sum(p.numel() for p in restored.parameters()))
        print(dict(variant=variant,status='passed',restore_exact=True),flush=True)
        del restored,opt2,sch2,saved,state,adam,a,b,uniform,noise,chain;torch.cuda.empty_cache()
    return dict(status='passed',rows=rows,crop_augmentation_matches_legacy=True,source_sha256=source_hashes(ROOT),
                scope='isolated engineering weights,3updates per arm; discarded,not effect training')


def benchmark_train(c):
    configure(c);model=ex.build(c).train();chain=ex.chain_for(c)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    dataset=ConditionDataset(c['manifest'],c['condition_cache'],'train',512);epoch=1;iterator=iter(ex.loader(dataset,c,epoch))
    def step():
        nonlocal iterator,epoch
        try:values=next(iterator)
        except StopIteration:epoch+=1;iterator=iter(ex.loader(dataset,c,epoch));values=next(iterator)
        image,condition,target=[x.cuda() for x in values[:3]]
        train_step(model,chain,optimizer,image,condition,target,c)
    for _ in range(10):step()
    synchronize('cuda');torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    for _ in range(30):step()
    synchronize('cuda');elapsed=time.perf_counter()-begin
    return dict(status='passed',mode=c['mode'],batch=c['batch_size'],warmup_updates=10,timed_updates=30,
                seconds_per_update=elapsed/30,samples_per_second=30*c['batch_size']/elapsed,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                total_memory_mib=torch.cuda.get_device_properties(0).total_memory/1024**2,
                scope='actual cache IO/augmentation/conditions/categorical teacher distribution/loss/backward/clip/AdamW; prior generated once separately; fresh effect initialization')


def benchmark_window(c,cold=False):
    configure(c);model=ex.build(c).eval();chain=ex.chain_for(c)
    dataset=ConditionDataset(c['manifest'],c['condition_cache'],'train',crop=None)
    begin=time.perf_counter();image,conditions,target,ident=dataset[0];read_seconds=time.perf_counter()-begin
    # Warm the actual new model without evaluating any labels.
    for _ in range(3):
        sample(model,chain,image[None,:,:512,:512].cuda(),conditions[None,:,:512,:512].cuda(),steps=5)
    prior_seconds=None;condition_seconds=None;cold_read_seconds=None
    if cold:
        teacher=load_direct(c['prior_checkpoint'],'cuda');begin=time.perf_counter()
        with Image.open(Path(dataset.manifest['dataset_root'])/dataset.manifest['image_pattern'].format(id=ident)) as im:raw=np.asarray(im.convert('RGB')).copy()
        image=(torch.from_numpy(raw).permute(2,0,1).float()/255-MEAN)/STD
        cold_read_seconds=time.perf_counter()-begin
        synchronize('cuda');begin=time.perf_counter()
        _,calls,prior_timing,p=prior_predict(teacher,None,image,ident,dict(c,mode='S0',window_batch_size=1),True)
        synchronize('cuda');prior_seconds=time.perf_counter()-begin
        begin=time.perf_counter();prior=((1-4e-4)*p.numpy()+1e-4).astype(np.float16)
        conditions=torch.from_numpy(conditions_for_region(raw,prior,substrate(raw,prior)))
        condition_seconds=time.perf_counter()-begin
    synchronize('cuda');torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    _,nfe,timing,_=ex.predict(model,chain,image,conditions,ident,c)
    synchronize('cuda');seconds=time.perf_counter()-begin
    return dict(status='passed',mode=c['mode'],window_batch=c['window_batch_size'],image_id=ident,
                source_split='train',new_model_native_seconds=seconds,read_and_cached_condition_seconds=read_seconds,
                cold_image_read_seconds=cold_read_seconds,cold_prior_inference_seconds=prior_seconds,cold_condition_preparation_seconds=condition_seconds,
                combined_cold_native_seconds=None if not cold else cold_read_seconds+prior_seconds+condition_seconds+seconds,
                nfe=nfe,prior_nfe=35,**timing,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                total_memory_mib=torch.cuda.get_device_properties(0).total_memory/1024**2,
                scope='engineering native-image timing, initialized model; cold prior separately measured; no metric or model selection')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify_cuda','benchmark_train','benchmark_window','benchmark_cold','train','secondary'])
    p.add_argument('--variant',choices=['N0','N1','N2','N3'],default='N0');p.add_argument('--batch',type=int,default=2);p.add_argument('--window-batch',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path)
    args=p.parse_args();c=make_config(args.variant,args.batch,args.window_batch)
    try:
        if args.action=='verify_cuda':result=verify_cuda()
        elif args.action=='benchmark_train':result=benchmark_train(c)
        elif args.action in ['benchmark_window','benchmark_cold']:result=benchmark_window(c,args.action=='benchmark_cold')
        else:
            freeze=json.loads(args.plan.read_text());c=freeze['configs'][args.variant];ex.guard_inputs(c,freeze);run=Path(freeze['runs'][args.variant])
            if args.action=='train':ex.train(c,freeze,run);result=dict(status='completed',run=str(run))
            else:
                configure(c);model=ex.build(c);state=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)
                assert state['config']==c and state['source_sha256']==freeze['source_sha256']
                assert state['prior_sha256']==freeze['prior_sha256'] and state['condition_metadata_sha256']==freeze['condition_metadata_sha256']
                model.load_state_dict(state['model']);c=dict(c,sampling_seed=3408)
                metrics,matrices,_,timing=ex.evaluate(model,ex.chain_for(c),c)
                np.savez_compressed(args.output.with_suffix('.npz'),**matrices);write_json(args.output.with_name(args.output.stem+'_timing.json'),timing)
                result=dict(status='completed',epoch=state['epoch'],sampling_seed=3408,metrics=metrics,checkpoint_sha256=digest(run/'checkpoints/best.pt'),purpose='same3407best, no reselection or new training')
        result['source_sha256']=source_hashes(ROOT)
        result['worker_sha256']=digest(__file__)
        write_json(args.output,result)
    except torch.cuda.OutOfMemoryError as exc:
        write_json(args.output,dict(status='oom',action=args.action,mode=c['mode'],batch=c['batch_size'],window_batch=c['window_batch_size'],error=str(exc),traceback=traceback.format_exc()))
        if args.action not in ['benchmark_train','benchmark_window']:raise
    except BaseException as exc:
        write_json(args.output,dict(status='failed',error=str(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()

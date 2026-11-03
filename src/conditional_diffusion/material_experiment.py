"""Isolated100-new-epoch experiment; historical source and guards are untouched."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .material_conditions import ConditionDataset, digest
from .material_categorical import MaterialModel,CategoricalChain,sample
from .material_categorical import train_step as material_step
from .data import read_manifest
from .inference import positions,image_seed,preview
from .metrics_impl import confusion,summarize,boundary_counts,summarize_boundaries
from .runtime import (write_json,source_hashes,StorageGuard,directory_size,
                      atomic_checkpoint,ExperimentLogger,rng_state,restore_rng)
from .static_experiment import configure,synchronize,scheduler_for,restore_best_sidecars


def config_for(root,variant,batch=2,window_batch=1):
    root=Path(root)
    c=json.loads((root/'configs/pilot_static_semantic_s2.json').read_text())
    c.update(schema_version=3,mode=variant,experiment_variant='material_categorical_'+variant.lower(),
             architecture='static_semantic_optical_conditions_v1',epochs=100,val_every=20,
             batch_size=batch,window_batch_size=window_batch,workers=4,timesteps=100,
             timestep_sampling='categorical_uniform1_to100',training_state='fixed_prior_categorical',
             recycle_fraction=0.,high_noise_start=None,kl_weight=1.,
             prior_checkpoint=str(root/'outputs/P1/direct/seed_42/p1_cosine_20261017/checkpoints/best.pt'),
             condition_cache=str(root/'outputs/launches/p1_material_categorical_20261020/conditions'),
             timestep_grid=[100,75,50,25,1],inherited_prior_training_epochs=40,
             initialization='same historical Direct RGB/decoder/head; matched new layers seed42; reset optimizer',
             conditioning='RGB + contrast3 + reliability1 + shared frozen prior4',
             final_readout='clean class probabilities',state_mapping='categorical_onehot_encoded_pm1',
             cache_storage='whole-image priorfloat16; RGB-only substrate metadata; train-only statistics')
    # Remove unrelated historical mode switches to avoid misleading metadata.
    c.pop('static_initialization_seed',None)
    return c


def loader(dataset,c,epoch):
    dataset.epoch=epoch
    return DataLoader(dataset,batch_size=c['batch_size'],shuffle=True,
                      generator=torch.Generator().manual_seed(c['seed']+epoch),
                      num_workers=c['workers'],pin_memory=c['device']=='cuda',drop_last=False)


def build(c):return MaterialModel(c['mode'],c['prior_checkpoint'],c['seed']).to(c['device'])
def chain_for(c):return CategoricalChain(str(Path(c['condition_cache'])/'optical_statistics.json'),c['device'],c['timesteps'])


@torch.no_grad()
def predict(model,chain,image,conditions,ident,c,return_probabilities=False):
    device=next(model.parameters()).device;h,w=image.shape[-2:]
    size,stride,batch=c['crop_size'],c['stride'],c['window_batch_size']
    if min(h,w)<size:raise ValueError('Unexpected small native image')
    clock=time.perf_counter();coords=[(y,x) for y in positions(h,size,stride) for x in positions(w,size,stride)]
    stochastic=model.variant in ['N2','N3'];steps=c['sampling_steps']
    # Each whole-image uniform field is fixed by image+stage; windows share it.
    fields=None
    if stochastic:
        generator=torch.Generator().manual_seed(image_seed(ident,c['sampling_seed']))
        fields=torch.rand((steps,h,w),generator=generator)
    total=torch.zeros((4,h,w));counts=torch.zeros((h,w))
    timing=dict(preparation_seconds=time.perf_counter()-clock,transfer_seconds=0.,sampling_seconds=0.,fusion_seconds=0.)
    for start in range(0,len(coords),batch):
        part=coords[start:start+batch];clock=time.perf_counter()
        rgb=torch.stack([image[:,y:y+size,x:x+size] for y,x in part]).to(device)
        aux=torch.stack([conditions[:,y:y+size,x:x+size] for y,x in part]).to(device)
        uniform=None if fields is None else torch.stack([fields[:,y:y+size,x:x+size] for y,x in part]).to(device)
        synchronize(device);timing['transfer_seconds']+=time.perf_counter()-clock;clock=time.perf_counter()
        probabilities=sample(model,chain,rgb,aux,uniform,steps)
        synchronize(device);timing['sampling_seconds']+=time.perf_counter()-clock;clock=time.perf_counter()
        probabilities=probabilities.cpu();synchronize(device)
        timing['transfer_seconds']+=time.perf_counter()-clock;clock=time.perf_counter()
        for p,(y,x) in zip(probabilities,part):total[:,y:y+size,x:x+size]+=p;counts[y:y+size,x:x+size]+=1
        timing['fusion_seconds']+=time.perf_counter()-clock
    assert bool((counts>0).all());clock=time.perf_counter();probabilities=total/counts
    prediction=probabilities.argmax(0).numpy().astype(np.uint8);timing['fusion_seconds']+=time.perf_counter()-clock
    calls=len(coords)*(1 if model.variant=='N0' else (5 if model.variant=='N1' else steps))
    return prediction,calls,timing,probabilities if return_probabilities else None


@torch.no_grad()
def evaluate(model,chain,c):
    dataset=ConditionDataset(c['manifest'],c['condition_cache'],'val',crop=None)
    model.eval();synchronize(c['device']);torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter();matrices=[];ids=[];breakdown=[];previews={};boundaries=np.zeros((4,4),np.int64);calls=0
    prior_reference=0.
    for index in range(len(dataset)):
        clock=time.perf_counter();image,conditions,target,ident=dataset[index];read=time.perf_counter()-clock
        predicted,nfe,timing,_=predict(model,chain,image,conditions,ident,c)
        reference=dataset.index['images'][ident]['prior_timing']
        prior_seconds=sum(reference[k] for k in ['preparation_seconds','transfer_seconds','sampling_seconds','fusion_seconds'])
        prior_reference+=prior_seconds
        clock=time.perf_counter();truth=target.numpy();matrices.append(confusion(truth,predicted))
        boundaries+=boundary_counts(truth,predicted,c['boundary_tolerance'])
        timing.update(id=ident,read_and_cached_condition_seconds=read,metrics_seconds=time.perf_counter()-clock,
                      frozen_prior_reference_inference_seconds=prior_seconds,
                      prior_accounting='Prior prepared once, cached in validation; reference is separately measured, not current walltime')
        breakdown.append(timing);ids.append(ident);calls+=nfe
        if index<c['preview_count']:previews[ident]=preview(image,truth,predicted)
    matrix=np.sum(matrices,axis=0);metrics={**summarize(matrix),**summarize_boundaries(boundaries)}
    metrics.update(seconds=time.perf_counter()-begin,image_count=len(ids),nfe_total=calls,
                   frozen_prior_reference_seconds=prior_reference,frozen_prior_nfe_total=875,
                   peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                   peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    return metrics,dict(ids=np.array(ids),per_image=np.array(matrices),global_matrix=matrix,boundary_counts=boundaries),previews,breakdown


def guard_inputs(c,freeze):
    root=Path(c['project'])
    assert source_hashes(root)==freeze['source_sha256'],'Material source/config freeze mismatch'
    assert digest(c['prior_checkpoint'])==freeze['prior_sha256']
    for p,h in freeze['condition_metadata_sha256'].items():assert digest(root/p)==h,p
    read_manifest(c['manifest'])


def checkpoint_payload(model,optimizer,scheduler,c,freeze,epoch,step,best):
    return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                config=c,source_sha256=freeze['source_sha256'],freeze_id=freeze['batch_id'],
                prior_sha256=freeze['prior_sha256'],condition_metadata_sha256=freeze['condition_metadata_sha256'],
                rng=rng_state(),epoch=epoch,global_step=step,best_miou=best)


def train(c,freeze,run):
    configure(c);run=Path(run);guard_inputs(c,freeze)
    assert c['seed']==42 and c['epochs']==100 and c['crop_size']==512 and c['val_every']==20
    assert c['sampling_steps']==5 and c['sampling_seed']==3407
    if run.exists() and not (run/'checkpoints/last.pt').exists():
        assert not (run/'metrics_steps.csv').exists() and not (run/'checkpoints/best.pt').exists()
        if (run/'config.json').exists():assert json.loads((run/'config.json').read_text())==c
        run.rename(run.with_name(run.name+'_incomplete_'+str(time.time_ns())))
    resume=(run/'checkpoints/last.pt').exists()
    model=build(c);chain=chain_for(c);parameters=sum(p.numel() for p in model.parameters())
    guard=StorageGuard(c['project'],run,c['storage']);guard.check(16*parameters+64*1024**2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);scheduler=scheduler_for(optimizer,c)
    epoch0=step=0;best=-float('inf')
    if resume:
        state=torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        assert state['config']==c and state['source_sha256']==freeze['source_sha256']
        assert state['prior_sha256']==freeze['prior_sha256'] and state['condition_metadata_sha256']==freeze['condition_metadata_sha256']
        for file in ['last.pt.tmp','best.pt.tmp']:
            pending=run/'checkpoints'/file
            if pending.exists():pending.rename(pending.with_name(pending.name+'.uncommitted_'+str(time.time_ns())))
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);restore_rng(state['rng'])
        epoch0,step,best=state['epoch'],state['global_step'],state['best_miou']
        if (run/'checkpoints/best.pt').exists():
            selected=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)
            assert selected['config']==c and selected['source_sha256']==freeze['source_sha256']
            assert selected['prior_sha256']==freeze['prior_sha256'] and selected['condition_metadata_sha256']==freeze['condition_metadata_sha256']
            restore_best_sidecars(run,selected);best=max(best,selected['validation_metrics']['miou_4class'])
    else:
        run.mkdir(parents=True,exist_ok=False);(run/'checkpoints').mkdir();write_json(run/'config.json',c)
        write_json(run/'run_metadata.json',dict(parameter_count=parameters,trainable_parameter_count=parameters,
                   frozen_prior_parameters=freeze['prior_parameters'],full_system_parameters=parameters+freeze['prior_parameters'],
                   planned_updates=100*math.ceil(120/c['batch_size']),new_exposures=12000,train_records=120,train_unique_rgb=119,
                   source_sha256=freeze['source_sha256'],prior_sha256=freeze['prior_sha256'],condition_metadata_sha256=freeze['condition_metadata_sha256'],
                   initialization=c['initialization'],budget='shared completedDirect40 + new100; not scratch100',
                   precision='FP32; deterministic; matmulTF32false/cuDNNTF32true',gpu=torch.cuda.get_device_name(),torch_version=torch.__version__))
        atomic_checkpoint(run/'checkpoints/last.pt',checkpoint_payload(model,optimizer,scheduler,c,freeze,0,0,best),guard)
    if epoch0==100:
        assert step==100*math.ceil(120/c['batch_size']);write_json(run/'status.json',dict(status='completed',last_completed_epoch=100,global_step=step));return
    dataset=ConditionDataset(c['manifest'],c['condition_cache'],'train',crop=512)
    assert len(dataset)==120
    logger=ExperimentLogger(run,guard,step if resume else None,epoch0 if resume else None)
    write_json(run/'status.json',dict(status='running',last_completed_epoch=epoch0,global_step=step))
    try:
        for epoch in range(epoch0+1,101):
            guard_inputs(c,freeze);model.train();synchronize(c['device']);torch.cuda.reset_peak_memory_stats()
            begin=time.perf_counter();seen=0;sums={};tallies={};lr=optimizer.param_groups[0]['lr']
            for image,conditions,target,_ in loader(dataset,c,epoch):
                image,conditions,target=[x.to(c['device']) for x in [image,conditions,target]]
                terms,grad,tally=material_step(model,chain,optimizer,image,conditions,target,c)
                n=len(image);seen+=n;step+=1
                for key,value in terms.items():sums[key]=sums.get(key,0)+n*value
                for key,value in tally.items():tallies[key]=tallies.get(key,0)+value
                if step%c['log_every']==0:logger.scalars('steps',{'train_step/'+k:v for k,v in dict(**terms,grad_norm_pre_clip=grad,lr=lr).items()},epoch,step)
            synchronize(c['device']);seconds=time.perf_counter()-begin
            assert seen==120 and step==epoch*math.ceil(120/c['batch_size'])
            scalars={'train_epoch/'+k:v/seen for k,v in sums.items()}
            scalars.update({'resources/train_seconds':seconds,'resources/train_samples_per_second':seen/seconds,
                            'resources/train_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                            'resources/train_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
                            'optim/epoch_lr_group_0':lr,'budget/epoch_exposures':seen,'budget/optimizer_updates':math.ceil(120/c['batch_size'])})
            scalars.update({'categorical_train/'+k:v/seen for k,v in tallies.items()});logger.scalars('epochs',scalars,epoch,step)
            metrics=None
            if epoch%20==0:
                metrics,matrices,previews,timing=evaluate(model,chain,c)
                assert metrics['image_count']==25 and metrics['nfe_total']==(875 if c['mode']=='N0' else 4375)
                logger.scalars('val',{'val/'+k:v for k,v in metrics.items()},epoch,step);logger.images(previews,epoch)
                np.savez_compressed(run/'latest_val_confusion.npz',**matrices);write_json(run/f'validation_timing_epoch{epoch}.json',timing)
                if metrics['miou_4class']>best:
                    best=metrics['miou_4class'];selected=dict(model=model.state_dict(),config=c,source_sha256=freeze['source_sha256'],
                         prior_sha256=freeze['prior_sha256'],condition_metadata_sha256=freeze['condition_metadata_sha256'],
                         epoch=epoch,global_step=step,validation_metrics=metrics,validation_matrices=matrices)
                    atomic_checkpoint(run/'checkpoints/best.pt',selected,guard);restore_best_sidecars(run,selected)
            scheduler.step();logger.flush();atomic_checkpoint(run/'checkpoints/last.pt',checkpoint_payload(model,optimizer,scheduler,c,freeze,epoch,step,best),guard)
            write_json(run/'status.json',dict(status='completed' if epoch==100 else 'running',last_completed_epoch=epoch,global_step=step))
            print(dict(mode=c['mode'],epoch=epoch,global_step=step,train_loss=sums['loss_total']/seen,train_seconds=seconds,
                       val_miou=None if metrics is None else metrics['miou_4class']),flush=True)
    except BaseException as exc:
        write_json(run/'status.json',dict(status='interrupted',error=str(exc),global_step=step,recovery='frozen last checkpoint only'));raise
    finally:logger.close()

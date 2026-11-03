"""TRAIN-only fixed-crop, zero-optimization diagnostic for completed B.
Writes only to a new diagnostic directory. Preserves all frozen inputs.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json,hashlib,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
ROOT=Path('/root/autodl-tmp/ConditionalDiffusionSeg')
sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.reference_data import dataset_from_config
from conditional_diffusion.reference_adapter import FrozenReferenceAdapter
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.recycling import mixed_recycling_input
from conditional_diffusion.objectives import losses
OUT=ROOT/'outputs/analysis/b_zero_step_diagnostic_20261020'
LAUNCH=ROOT/'outputs/launches/p1_reference_adapter_20261018'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def model_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  h.update(k.encode());h.update(str(v.dtype).encode());h.update(np.asarray(v.shape,dtype=np.int64).tobytes());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def stats(x):
 x=x.detach().float()
 return {'mean_abs':x.abs().mean().item(),'rms':x.square().mean().sqrt().item(),'max_abs':x.abs().max().item()}
def main():
 assert not OUT.exists(),'Preserve old diagnostic directory'
 OUT.mkdir(parents=True)
 begin=time.monotonic()
 plan=json.loads((LAUNCH/'plan.json').read_text())
 frozen={}
 for key in ['source_sha256','protected_artifacts_sha256','workflow_sha256','preflight_artifacts_sha256']:
  for rel,expected in plan[key].items():
   path=ROOT/rel
   actual=sha(path)
   assert actual==expected,(key,rel)
   frozen[str(path)]=actual
 ref=json.loads((LAUNCH/'b_seed3408.json').read_text())
 checkpoint=Path(ref['checkpoint']); assert sha(checkpoint)==ref['checkpoint_sha256']
 config_path=checkpoint.parent.parent/'config.json'
 c=json.loads(config_path.read_text())
 assert c==plan['configs']['b']
 for p,expected in [(Path(c['adapter_base_checkpoint']),c['adapter_base_sha256']),
                    (Path(c['reference_cache']),c['reference_cache_sha256'])]:
  assert sha(p)==expected
 inputs={str(p):sha(p) for p in [checkpoint,config_path,Path(c['adapter_base_checkpoint']),Path(c['reference_cache']),Path(c['manifest']),Path(__file__),LAUNCH/'plan.json',LAUNCH/'b_seed3408.json']}
 torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 ds=dataset_from_config(c,'train',crop=512,seed=42,augment=True);ds.epoch=40
 indices=[i for i,k in enumerate(ds.ids) if ds.references[k]['accepted']][:8]
 assert len(indices)==8
 items=[ds[i] for i in indices]
 crops=[]
 for i,(image,mask,ident) in zip(indices,items):
  rgb,_,_=ds.raw(i);rng=np.random.default_rng(np.random.SeedSequence([42,40,i]))
  y=int(rng.integers(rgb.shape[0]-512+1));x=int(rng.integers(rgb.shape[1]-512+1));k=int(rng.integers(4));flips=[bool(rng.random()<.5) for _ in [0,1]]
  for pattern in ['image_pattern','mask_pattern']:
   p=ds.root/ds.manifest[pattern].format(id=ident);inputs[str(p)]=sha(p)
  crops.append({'id':ident,'train_index':i,'original_xyxy':[x,y,x+512,y+512],'rotation_k90':k,'flip_axis0_axis1':flips,'q':image[6,0,0].item(),'target_pixel_counts':torch.bincount(mask.flatten(),minlength=4).tolist(),'image_tensor_sha256':hashlib.sha256(image.numpy().tobytes()).hexdigest()})
 image_cpu=torch.stack([r[0] for r in items]);target_cpu=torch.stack([r[1] for r in items])
 permutation=[(i+1)%8 for i in range(8)]
 protocol={'scope':'TRAIN ONLY; no val/test image access, no optimizer construction, no updates, no weight reselection.', 'crop_selection':'First 8 accepted images in frozen train order, no GT/performance filtering. Original dataset deterministic crop+augmentation at seed42 epoch40.', 'crops':crops,'contrast_permutation_target_source':permutation,'contrast_variants':['real','zero','permuted'],'q_rgb_t_noisy_identical_across_variants':True,'batch_size':2,'noise_seed':20261020,'timesteps_per_batch':[[100,900],[300,933],[500,966],[700,999]],'training_state':'One real-C mixed_recycling teacher computation per batch, saved detached noisy state reused in all three C interventions. This isolates direct C sensitivity; not resampling teacher under each intervention.','loss_weighting':{'CE':.5,'Dice_fg':.5,'MSE':1.},'gradient_scope':'adapter parameters only; unweighted per-term vectors plus weighted norm/cosine; no clipping, no optimizer.'}
 write('protocol.json',protocol)
 st=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert st['config']==c
 model=FrozenReferenceAdapter(c).cuda();model.load_state_dict(st['model'],strict=True);model.eval()
 del st
 before=model_sha(model)
 assert all(not p.requires_grad for p in model.base.parameters())
 parameters=list(model.adapter.parameters());diff=GaussianDiffusion(1000,'cuda')
 captures={}
 hooks=[model.base.register_forward_hook(lambda m,a,o:captures.update(base=o.detach())),model.adapter.register_forward_hook(lambda m,a,o:captures.update(raw=o.detach()))]
 torch.manual_seed(20261020);torch.cuda.manual_seed_all(20261020)
 records=[];aggregate={v:{k:torch.zeros(sum(p.numel() for p in parameters)) for k in ['CE','Dice_fg','MSE']} for v in protocol['contrast_variants']}
 for bi in range(4):
  sl=slice(2*bi,2*bi+2);im=image_cpu[sl].cuda();target=target_cpu[sl].cuda();x0=diff.encode(target)
  t=torch.tensor(protocol['timesteps_per_batch'][bi],device='cuda');noise=torch.randn_like(x0)
  noisy,recycled=mixed_recycling_input(model,diff,im,x0,t,noise,.5)
  batch={'batch':bi,'ids':[items[j][2] for j in range(2*bi,2*bi+2)],'t':t.tolist(),'recycled_indices':recycled.tolist(),'noisy_sha256':hashlib.sha256(noisy.cpu().numpy().tobytes()).hexdigest(),'variants':{}}
  outputs={}
  for variant in protocol['contrast_variants']:
   intervention=im.clone()
   if variant=='zero':intervention[:,3:6]=0
   elif variant=='permuted':intervention[:,3:6]=image_cpu[permutation[2*bi:2*bi+2],3:6].cuda()
   assert torch.equal(intervention[:,:3],im[:,:3]) and torch.equal(intervention[:,6:],im[:,6:])
   out=model(intervention,noisy,t);base=captures['base'];raw=captures['raw'];correction=im[:,6:7]*raw
   assert torch.allclose(out.detach()-base,correction,atol=1e-6,rtol=1e-5)
   total,terms=losses(out,target,c,x0)
   vectors={}
   for name,key in [('CE','loss_ce'),('Dice_fg','loss_dice_fg'),('MSE','loss_x0_mse')]:
    grad=torch.autograd.grad(terms[key],parameters,retain_graph=True,allow_unused=False)
    vec=torch.cat([g.detach().flatten() for g in grad]).cpu();assert torch.isfinite(vec).all();vectors[name]=vec
    aggregate[variant][name]+=vec/4
   weighted={k:v*protocol['loss_weighting'][k] for k,v in vectors.items()}
   cos={a+'__'+b:torch.nn.functional.cosine_similarity(vectors[a][None],vectors[b][None]).item() for a,b in [('CE','Dice_fg'),('CE','MSE'),('Dice_fg','MSE')]}
   value={'losses':{k:v.detach().item() for k,v in terms.items()},'raw_correction':stats(raw),'q_correction':stats(correction),'base_output':stats(base),'q_correction_rms_over_base_rms':stats(correction)['rms']/max(stats(base)['rms'],1e-12),'grad_unweighted_norm':{k:v.norm().item() for k,v in vectors.items()},'grad_weighted_norm':{k:v.norm().item() for k,v in weighted.items()},'grad_cosine':cos,'grad_weighted_total_norm':sum(weighted.values()).norm().item()}
   outputs[variant]=out.detach();batch['variants'][variant]=value
   del out,total,terms,vectors,weighted
  batch['sensitivity']={}
  for a,b in [('real','zero'),('real','permuted'),('zero','permuted')]:
   oa,ob=outputs[a],outputs[b];delta=oa-ob
   batch['sensitivity'][a+'__'+b]={**stats(delta),'argmax_changed_fraction':(oa.argmax(1)!=ob.argmax(1)).float().mean().item(),'softmax_mean_abs_change':(oa.softmax(1)-ob.softmax(1)).abs().mean().item()}
  assert all(p.grad is None for p in model.base.parameters())
  assert all(p.grad is None for p in model.adapter.parameters())
  records.append(batch);write('partial.json',{'completed_batches':len(records),'batches':records})
  print('completed batch',bi,flush=True)
  del outputs,im,target,x0,noisy
 with torch.no_grad():
  rejected=image_cpu[:2].cuda();rejected[:,6]=0
  check=model(rejected,noise,t)
  bypass_exact=torch.equal(check,captures['base'])
 assert bypass_exact
 after=model_sha(model);assert before==after
 assert sha(checkpoint)==ref['checkpoint_sha256']
 assert all(sha(Path(p))==s for p,s in inputs.items())
 assert all(sha(Path(p))==s for p,s in frozen.items())
 for h in hooks:h.remove()
 gradients={}
 for variant,vs in aggregate.items():
  gradients[variant]={'mean_batch_gradient_unweighted_norm':{k:v.norm().item() for k,v in vs.items()},'mean_batch_gradient_cosine':{a+'__'+b:torch.nn.functional.cosine_similarity(vs[a][None],vs[b][None]).item() for a,b in [('CE','Dice_fg'),('CE','MSE'),('Dice_fg','MSE')]}}
 summary={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'status':'completed','elapsed_seconds':time.monotonic()-begin,'original_screening_remains_failed':True,'optimizer_steps':0,'optimizer_constructed':False,'checkpoint_epoch':ref['checkpoint_epoch'],'checkpoint_sha256_before_after':ref['checkpoint_sha256'],'model_state_sha256_before':before,'model_state_sha256_after':after,'frozen_files_verified_before_after':len(frozen),'base_grad_all_none':all(p.grad is None for p in model.base.parameters()),'base_requires_grad_all_false':all(not p.requires_grad for p in model.base.parameters()),'adapter_grad_buffers_all_none':all(p.grad is None for p in model.adapter.parameters()),'q_zero_bypass_exact':bypass_exact,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in parameters),'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'protocol':protocol,'aggregate_gradients':gradients,'batches':records,'input_sha256':inputs,'frozen_sha256':frozen,'interpretation_limits':['Training crops at four fixed timestep pairs are diagnostic only, not validation/test evidence.','Real-vs-zero/permuted C effects isolate this checkpoint and these held-fixed noisy states; not causal proof of failure source.','Nonzero gradients do not imply improvement from additional training; no such training is authorized in this diagnostic.','Cosines describe instantaneous adapter parameter gradients; no statistical significance or generalization claim.','Original A/B failed screening remains unchanged; no rescue, unfreeze, extension, or new reference training.']}
 write('summary.json',summary)
 print('DONE',summary['elapsed_seconds'],flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  if OUT.exists():(OUT/'failure.txt').write_text(traceback.format_exc())
  raise

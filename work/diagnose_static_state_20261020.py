"""Existing S1/S2 TRAIN-only zero-optimization state/objective diagnostic."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,hashlib,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch.nn import functional as F
R=Path('/root/autodl-tmp/ConditionalDiffusionSeg');sys.path.insert(0,str(R/'src'))
from conditional_diffusion.data import MaterialDataset
from conditional_diffusion.static_semantic import StaticSemanticModel
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.objectives import losses
from conditional_diffusion.recycling import mixed_recycling_input
O=R/'outputs/analysis/static_state_diagnostic_20261020';L=R/'outputs/launches/p1_static_semantic_20261020';TS=[0,250,500,750,999]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,d):(O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stat(a):
 a=a.detach().float();return dict(mean=a.mean().item(),rms=a.square().mean().sqrt().item(),min=a.min().item(),max=a.max().item())
def ms(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def metrics(cm):
 c=np.asarray(cm);den=c.sum(0)+c.sum(1)-np.diag(c);ious=np.divide(np.diag(c),den,out=np.zeros(4,dtype=float),where=den>0)
 return dict(matrix=c.tolist(),iou=ious.tolist(),thin_iou=float(ious[1:3].mean()),mono_to_background=int(c[1,0]),few_to_mono=int(c[2,1]),mono_to_background_rate=float(c[1,0]/max(1,c[1].sum())),few_to_mono_rate=float(c[2,1]/max(1,c[2].sum())))
def confusion(y,p):return np.bincount((y.astype(np.int64)*4+p).ravel(),minlength=16).reshape(4,4)
def main():
 assert not O.exists(),'Preserve completed diagnostics';O.mkdir(parents=True);begin=time.monotonic()
 plan=json.loads((L/'plan.json').read_text());frozen={}
 for key in ['source_sha256','workflow_sha256']:
  for p,s in plan[key].items():assert sha(R/p)==s,(key,p);frozen[str(R/p)]=s
 inputs={str(L/'plan.json'):sha(L/'plan.json'),str(Path(__file__)):sha(__file__)}
 ds=MaterialDataset(plan['configs']['S2']['manifest'],'train',crop=512,seed=42,augment=True);ds.epoch=40
 candidates=[]
 for i in range(len(ds)):
  im,gt,ident=ds[i];counts=torch.bincount(gt.flatten(),minlength=4).tolist();candidates.append((i,ident,counts))
 selected=[z for z in candidates if z[2][1]>=1024][:4]
 selected += [z for z in candidates if z[2][2]>=1024 and z[0] not in [v[0] for v in selected]][:4]
 assert len(selected)==8
 images=[];targets=[];crops=[]
 for i,ident,counts in selected:
  im,gt,_=ds[i];images.append(im);targets.append(gt)
  raw,_,_=ds.raw(i);rng=np.random.default_rng(np.random.SeedSequence([42,40,i]));y=int(rng.integers(raw.shape[0]-511));x=int(rng.integers(raw.shape[1]-511));k=int(rng.integers(4));fl=[bool(rng.random()<.5) for _ in [0,1]]
  crops.append(dict(id=ident,train_index=i,xyxy=[x,y,x+512,y+512],rotation_k90=k,flip_axes=fl,pixel_counts=counts))
  for p in ['image_pattern','mask_pattern']:
   f=ds.root/ds.manifest[p].format(id=ident);inputs[str(f)]=sha(f)
 image=torch.stack(images);target=torch.stack(targets);x0cpu=GaussianDiffusion.encode(target)
 gen=torch.Generator().manual_seed(20261020);noise_cpu=torch.randn(x0cpu.shape,generator=gen)
 protocol=dict(scope='TRAIN ONLY, existing bests, zero optimizer construction/steps, no val/test image access or weight selection',selection='Seed42 epoch40 deterministic crop+augmentation. First4 train-order crops with monolayer>=1024 pixels, then first4 remaining with fewlayer>=1024 pixels; GT-only diagnostic coverage, no model scores.',crops=crops,main_resolution=512,gradient_resolution=64,gradient_downsample='RGB bilinear align_cornersFalse; masks nearest; x0 re-encoded after downsample; not full-resolution gradient evidence.',timesteps=TS,noise_seed=20261020,state_permutation=[(i+1)%8 for i in range(8)],time_permutation=[250,500,750,999,0],intervention_scope='S2 q_sample(encodedGT,t,one fixed Gaussian per crop). RGB/noise/t held fixed for state interventions; zero/permuted time hold same noisy state. These include intentionally mismatched/OOD controls, not alternate evaluation protocols.',gradient_scope='Same64 input and raw logits; CE, foregroundDice, rawx0MSE, MSE(2softmax(logits)-1,encodedGT). All parameters, named RGB/state-time/decoder groups; no clipping/updates.',trajectories='Native512, cachedRGB, actual five-step rounded999,749,500,250,0; S2 pure Gaussian, clippedx0 eta0DDIM; S1 uniformencodedstate and softmaxencode updates. Crop-only, no full-image stitching.')
 write('protocol.json',protocol)
 torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 diff=GaussianDiffusion(1000,'cuda');models={};before={};checkpoint_info={}
 for mode in ['S1','S2']:
  cp=Path(plan['runs'][mode])/'checkpoints/best.pt';config=cp.parent.parent/'config.json';inputs[str(cp)]=sha(cp);inputs[str(config)]=sha(config)
  st=torch.load(cp,map_location='cpu',weights_only=False);assert st['config']==plan['configs'][mode];assert st['source_sha256']==plan['source_sha256']
  model=StaticSemanticModel(mode).cuda().eval();model.load_state_dict(st['model'],strict=True);models[mode]=model;before[mode]=ms(model);checkpoint_info[mode]=dict(path=str(cp),sha256=inputs[str(cp)],epoch=st['epoch']);del st
 print('GPU_STARTED',flush=True)
 sensitivities=[];s2=models['S2']
 with torch.no_grad():
  for i in range(8):
   im=image[i:i+1].cuda();feat=s2.encode_rgb(im);x0=x0cpu[i:i+1].cuda();eps=noise_cpu[i:i+1].cuda();other=x0cpu[(i+1)%8:(i+1)%8+1].cuda();oeps=noise_cpu[(i+1)%8:(i+1)%8+1].cuda()
   for j,tv in enumerate(TS):
    t=torch.tensor([tv],device='cuda');state=diff.q_sample(x0,t,eps);permuted=diff.q_sample(other,t,oeps);real=s2.forward_with_rgb(feat,state,t)
    row=dict(crop=i,t=tv,state=stat(state),output=stat(real),variants={})
    for name,ss,tt in [('zero_state',torch.zeros_like(state),t),('permuted_state',permuted,t),('zero_time',state,torch.zeros_like(t)),('permuted_time',state,torch.tensor([protocol['time_permutation'][j]],device='cuda'))]:
     out=s2.forward_with_rgb(feat,ss,tt);delta=out-real
     row['variants'][name]=dict(mean_abs=delta.abs().mean().item(),rms=delta.square().mean().sqrt().item(),argmax_changed_fraction=(out.argmax(1)!=real.argmax(1)).float().mean().item())
    sensitivities.append(row)
  write('sensitivity_512.json',sensitivities)
 print('SENSITIVITY_DONE',flush=True)
 trajectories={};predictions={}
 with torch.no_grad():
  for mode,model in models.items():
   cms=[np.zeros((4,4),dtype=np.int64) for _ in range(5)];change=[dict(corrected_vs_previous=0,new_errors_vs_previous=0,corrected_vs_first=0,new_errors_vs_first=0,ever_corrected=0,ever_new_error=0) for _ in range(5)];state_stats=[];raw_clip_changes=[]
   for i in range(8):
    im=image[i:i+1].cuda();feat=model.encode_rgb(im);y=target[i].numpy();state=noise_cpu[i:i+1].cuda() if mode=='S2' else torch.full((1,4,512,512),-.5,device='cuda');first=None;prev=None;ever_c=np.zeros_like(y,dtype=bool);ever_e=ever_c.copy()
    for j,tv in enumerate(model.stage_timesteps.tolist()):
     state_stats.append(dict(crop=i,step=j,t=tv,state=stat(state)));t=torch.tensor([tv],device='cuda');raw=model.forward_with_rgb(feat,state,t);clean=raw.clamp(-1,1) if mode=='S2' else raw;pred=clean.argmax(1)[0].cpu().numpy();correct=pred==y;cms[j]+=confusion(y,pred);predictions[f'{mode}_{i}_{j}']=pred.astype(np.uint8)
     raw_clip_changes.append(dict(crop=i,step=j,fraction=(raw.argmax(1)!=clean.argmax(1)).float().mean().item()))
     if j==0:first=correct.copy()
     else:
      cc=(~prev)&correct;ee=prev&(~correct);ever_c|=cc;ever_e|=ee;change[j]['corrected_vs_previous']+=int(cc.sum());change[j]['new_errors_vs_previous']+=int(ee.sum())
     change[j]['corrected_vs_first']+=int(((~first)&correct).sum());change[j]['new_errors_vs_first']+=int((first&(~correct)).sum());change[j]['ever_corrected']+=int(ever_c.sum());change[j]['ever_new_error']+=int(ever_e.sum());prev=correct
     if mode=='S1':state=2*raw.softmax(1)-1
     elif j<4:
      a,ap=diff.abar[tv],diff.abar[model.stage_timesteps[j+1]];epsilon=(state-a.sqrt()*clean)/(1-a).sqrt();state=ap.sqrt()*clean+(1-ap).sqrt()*epsilon
   trajectories[mode]=dict(steps=[dict(step=j,t=int(model.stage_timesteps[j]),**metrics(cm),**change[j]) for j,cm in enumerate(cms)],state_statistics=state_stats,raw_vs_clipped_argmax=raw_clip_changes)
  write('trajectories_512.json',trajectories);np.savez_compressed(O/'trajectory_labels_512.npz',**predictions)
 print('TRAJECTORY_DONE',flush=True)
 # Full parameter gradients only at64, independent per crop/time.
 gradients=[]
 for mode,model in models.items():
  named=list(model.named_parameters());params=[p for _,p in named];groups={g:[j for j,(n,p) in enumerate(named) if (n.startswith('rgb_encoder') if g=='RGB' else n.startswith(('state_encoder','state_projection','time_mlp')) if g=='state_time' else n.startswith(('decoder','head')))] for g in ['RGB','state_time','decoder_head']}
  for i in range(8):
   im=F.interpolate(image[i:i+1].cuda(),size=(64,64),mode='bilinear',align_corners=False);y=F.interpolate(target[i:i+1,None].float().cuda(),size=(64,64),mode='nearest')[:,0].long();x0=diff.encode(y);eps=F.interpolate(noise_cpu[i:i+1].cuda(),size=(64,64),mode='nearest')
   for tv in (TS if mode=='S2' else [None]):
    if mode=='S2':t=torch.tensor([tv],device='cuda');state=diff.q_sample(x0,t,eps);raw=model(im,state,t)
    else:raw=model(im)
    _,terms=losses(raw,y,{'boundary':False},x0);terms2={'CE':terms['loss_ce'],'Dice':terms['loss_dice_fg'],'raw_MSE':terms['loss_x0_mse'],'prob_MSE':F.mse_loss(2*raw.softmax(1)-1,x0)}
    gs={k:torch.autograd.grad(v,params,retain_graph=True,allow_unused=True) for k,v in terms2.items()}
    result=dict(mode=mode,crop=i,t=tv,loss={k:v.item() for k,v in terms2.items()},raw=stat(raw),raw_channel_sum=stat(raw.sum(1)),groups={})
    for group,ix in {**groups,'all':list(range(len(params)))}.items():
     norm={k:sum(g[j].square().sum() for j in ix if g[j] is not None).sqrt() for k,g in gs.items()}
     cosine={}
     for a,b in [('CE','Dice'),('CE','raw_MSE'),('Dice','raw_MSE'),('CE','prob_MSE'),('Dice','prob_MSE'),('raw_MSE','prob_MSE')]:
      dot=sum(gs[a][j].mul(gs[b][j]).sum() for j in ix if gs[a][j] is not None and gs[b][j] is not None);cosine[a+'__'+b]=(dot/(norm[a]*norm[b]).clamp_min(1e-20)).item()
     result['groups'][group]=dict(norm={k:v.item() for k,v in norm.items()},cosine=cosine)
    gradients.append(result);del gs,raw,terms,terms2
   write('gradient_64_partial.json',gradients)
 write('gradients_64.json',gradients)
 after={k:ms(m) for k,m in models.items()};assert before==after
 assert all(p.grad is None for m in models.values() for p in m.parameters())
 assert all(sha(p)==s for p,s in inputs.items());assert all(sha(p)==s for p,s in frozen.items())
 summary=dict(status='completed',completed_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-begin,optimizer_constructed=False,optimizer_steps=0,checkpoint=checkpoint_info,in_memory_model_sha256_before=before,in_memory_model_sha256_after=after,all_parameter_grad_buffers_none=True,frozen_files_verified_before_after=len(frozen),input_sha256=inputs,frozen_sha256=frozen,protocol=protocol,gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,output_files=['sensitivity_512.json','trajectories_512.json','trajectory_labels_512.npz','gradients_64.json'],limits=['Native512 sensitivity/trajectory and64 gradients have different spatial statistics; gradient evidence does not directly establish full-resolution cause.','GT-noised diagnostic states omit the training recycling mixture; trajectory initial pure noise is actual sampler input.','Permuted/zero state/time are diagnostic interventions, not valid inference alternatives.','All examples are GT-coverage-selected training crops. No validation generalization claim.','Raw/probMSE comparison uses identical raw logits at each point; does not establish that changing objective improves quality.','Original platform screening remains failed.'])
 write('summary.json',summary);print('DONE',summary['elapsed_seconds'],flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():(O/'failure.txt').write_text(traceback.format_exc())
  raise

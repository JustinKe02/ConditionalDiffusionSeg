import os,sys,json,csv
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,numpy as np
from PIL import Image
from torch.utils.data import DataLoader
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.data import MaterialDataset
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.model import make_model
from conditional_diffusion.inference import image_seed
from conditional_diffusion.metrics_impl import confusion,summarize
from conditional_diffusion.objectives import losses
run=ROOT/'outputs/P1/diffusion/seed_42/p1_cosine_20261017';out=ROOT/'outputs/launches/p1_cosine_20261017/analysis'
state=torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False);c=state['config'];torch.set_num_threads(4)
result={'scope':'Read-only diagnosis of final P1 diffusion checkpoint. 12 training patches are a prespecified subset, not full-train metrics.'}
def save(): (out/'deep_diffusion_diagnosis.json').write_text(json.dumps(result,indent=2))
# Reconstruct actual timestep assignment and prove it against saved CUDA RNG.
torch.cuda.manual_seed_all(42)
ts=np.zeros((40,120),dtype=np.int64)
for epoch in range(1,41):
 loader=DataLoader(range(120),batch_size=2,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42+epoch))
 for indices in loader:
  t=torch.randint(1000,(2,),device='cuda');ts[epoch-1,indices.numpy()]=t.cpu().numpy()
  torch.randn((2,4,512,512),device='cuda')
rng_matches=torch.equal(torch.cuda.get_rng_state(),state['rng']['cuda'][0]);assert rng_matches
counts={str(v):int((ts>=v).sum()) for v in [900,950,990,999]}
result['timesteps']={'exact_cuda_rng_match':rng_matches,'counts':counts,'samples':int(ts.size),'per_image':{str(v):{'mean':float((ts>=v).sum(0).mean()),'minimum':int((ts>=v).sum(0).min()),'maximum':int((ts>=v).sum(0).max()),'images_with_zero':int(((ts>=v).sum(0)==0).sum())} for v in [900,950,990,999]}}
print('TIMESTEPS',json.dumps(result['timesteps']),flush=True);save()
# Read only train masks; reconstruct the actual 4800 crop coordinates/counts.
ds=MaterialDataset(c['manifest'],'train',crop=512,seed=42,augment=True)
counts_crop=np.zeros((40,120,4),dtype=np.int64)
for i,ident in enumerate(ds.ids):
 path=ds.root/ds.manifest['mask_pattern'].format(id=ident)
 with Image.open(path) as im: mask=np.asarray(im)
 for epoch in range(1,41):
  rng=np.random.default_rng(np.random.SeedSequence([42,epoch,i]));h,w=mask.shape
  y=int(rng.integers(h-512+1));x=int(rng.integers(w-512+1))
  counts_crop[epoch-1,i]=np.bincount(mask[y:y+512,x:x+512].ravel(),minlength=4)
result['training_crops']={'pixel_fractions':(counts_crop.sum((0,1))/counts_crop.sum()).tolist(),
 'fraction_of_crops_class_absent':(counts_crop==0).mean((0,1)).tolist(),
 'high_noise_crops_containing_class':{str(v):((counts_crop>0)&(ts>=v)[...,None]).sum((0,1)).tolist() for v in [900,950,990,999]}}
print('CROP_DISTRIBUTION',json.dumps(result['training_crops']),flush=True);save()
ds.epoch=40;indices=np.random.default_rng(42).permutation(120)[:12].tolist();patches=[ds[i] for i in indices]
model=make_model(c).cuda().eval();model.load_state_dict(state['model']);d=GaussianDiffusion(1000,'cuda')
result['rgb_ablations']=[]
with torch.no_grad():
 for tval in [0,500,950,999]:
  for mode in ['matched_rgb','zero_rgb','permuted_rgb']:
   matrix=np.zeros((4,4),dtype=np.int64)
   for i,(image,target,ident) in enumerate(patches):
    if mode=='zero_rgb':image=torch.zeros_like(image)
    elif mode=='permuted_rgb':image=patches[(i+1)%len(patches)][0]
    gen=torch.Generator().manual_seed(image_seed(ident,3407));noise=torch.randn((1,4,512,512),generator=gen).cuda()
    mask=target[None].cuda();t=torch.tensor([tval],device='cuda');xt=d.q_sample(d.encode(mask),t,noise)
    pred=model(image[None].cuda(),xt,t).argmax(1)[0].cpu().numpy();matrix+=confusion(target.numpy(),pred)
   item={'t':tval,'condition':mode,'metrics':summarize(matrix)};result['rgb_ablations'].append(item)
   print('RGB',tval,mode,item['metrics']['miou_4class'],flush=True);save()
 # Exact 20-step trajectory compared with same-noise GT-corrupted trajectory.
 grid=torch.linspace(999,0,20,device='cuda').round().long()
 mats={key:[np.zeros((4,4),dtype=np.int64) for _ in grid] for key in ['rollout','teacher_forced']}
 changed=np.zeros(len(grid));gt_values=np.zeros(len(grid));false_values=np.zeros(len(grid));sat=np.zeros(len(grid));drift=np.zeros(len(grid))
 for image,target,ident in patches:
  im=image[None].cuda();mask=target[None].cuda();x0=d.encode(mask)
  gen=torch.Generator().manual_seed(image_seed(ident,3407));noise=torch.randn((1,4,512,512),generator=gen).cuda();xt=noise.clone()
  correct=torch.nn.functional.one_hot(mask,4).permute(0,3,1,2).bool()
  for j,t in enumerate(grid):
   raw=model(im,xt,t.expand(1));clean=raw.clamp(-1,1);oracle=d.q_sample(x0,t.reshape(1),noise);teacher=model(im,oracle,t.expand(1))
   mats['rollout'][j]+=confusion(target.numpy(),raw.argmax(1)[0].cpu().numpy())
   mats['teacher_forced'][j]+=confusion(target.numpy(),teacher.argmax(1)[0].cpu().numpy())
   changed[j]+=float((raw.argmax(1)!=clean.argmax(1)).float().mean())/12
   gt_values[j]+=float(raw[correct].mean())/12;false_values[j]+=float(raw[~correct].mean())/12
   sat[j]+=float((raw.abs()>1).float().mean())/12;drift[j]+=float((xt-oracle).square().mean())/12
   if j<len(grid)-1:
    a,ap=d.abar[t],d.abar[grid[j+1]];eps=(xt-a.sqrt()*clean)/(1-a).sqrt();xt=ap.sqrt()*clean+(1-ap).sqrt()*eps
 result['trajectory']=[dict(t=int(t),alpha_bar=float(d.abar[t]),rollout=summarize(mats['rollout'][j]),teacher_forced=summarize(mats['teacher_forced'][j]),clip_changes_argmax_fraction=changed[j],mean_correct_class_output=gt_values[j],mean_other_class_output=false_values[j],output_outside_target_range_fraction=sat[j],state_vs_oracle_mse=drift[j]) for j,t in enumerate(grid)]
 # Gold +/-1 codes are perfect class predictions but not CE/Dice minima.
 terms_mean={}
 for start in range(0,12,2):
  masks=torch.stack([x[1] for x in patches[start:start+2]]).cuda();x0=d.encode(masks);_,terms=losses(x0,masks,c,x0)
  for key,val in terms.items():terms_mean[key]=terms_mean.get(key,0)+float(val)/6
 result['loss_on_exact_target_codes']=terms_mean
result['notes']=['Zero/permuted RGB are out-of-distribution ablations; they measure sensitivity, not legitimate inference protocols.','Clipping changes are measured along one trajectory on the fixed 12-patch subset; cannot rule out all amplitude/clipping effects.','Shared-head CE/MSE tension is a design risk, not a demonstrated cause without an independent training ablation.']
save();print('DONE',flush=True)

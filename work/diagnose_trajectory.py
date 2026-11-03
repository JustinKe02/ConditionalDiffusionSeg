import os,sys,json
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.data import FixedCropDataset
from conditional_diffusion.model import make_model
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.inference import image_seed
from conditional_diffusion.metrics_impl import confusion,summarize
run=ROOT/'outputs/P0/diffusion/seed_42/diffusion_resume_20261017'
out=ROOT/'outputs/launches/diffusion_resume_20261017/diagnostics'
torch.set_num_threads(4)
results={'trajectories':[]}
with torch.no_grad():
 for checkpoint in ['best','last']:
  state=torch.load(run/f'checkpoints/{checkpoint}.pt',map_location='cpu',weights_only=False)
  c=state['config'];model=make_model(c).cuda().eval();model.load_state_dict(state['model'])
  ds=FixedCropDataset(c['manifest'],c['fixed_crops']);d=GaussianDiffusion(1000,'cuda')
  grid=torch.linspace(999,0,20,device='cuda').round().long()
  for seed in [3407,3408]:
   mats={key:[np.zeros((4,4),dtype=np.int64) for _ in grid] for key in ['rollout_raw','rollout_clipped','teacher_same_noise']}
   drift=np.zeros(len(grid));changed=np.zeros(len(grid))
   for i in range(len(ds)):
    im,mask,ident=ds[i];im=im[None].cuda();gt=mask.numpy();x0=d.encode(mask[None].cuda())
    gen=torch.Generator().manual_seed(image_seed(ident,seed));noise=torch.randn((1,4,256,256),generator=gen).cuda();xt=noise.clone()
    for j,t in enumerate(grid):
     oracle=d.q_sample(x0,t.reshape(1),noise)
     raw=model(im,xt,t.expand(1));clean=raw.clamp(-1,1)
     teacher=model(im,oracle,t.expand(1))
     preds={'rollout_raw':raw.argmax(1)[0].cpu().numpy(),'rollout_clipped':clean.argmax(1)[0].cpu().numpy(),'teacher_same_noise':teacher.argmax(1)[0].cpu().numpy()}
     for key,pred in preds.items():mats[key][j]+=confusion(gt,pred)
     drift[j]+=float((xt-oracle).square().mean())/len(ds)
     changed[j]+=float((raw.argmax(1)!=clean.argmax(1)).float().mean())/len(ds)
     if j<len(grid)-1:
      a,ap=d.abar[t],d.abar[grid[j+1]]
      eps=(xt-a.sqrt()*clean)/(1-a).sqrt();xt=ap.sqrt()*clean+(1-ap).sqrt()*eps
   row={'checkpoint':checkpoint,'seed':seed,'steps':[{'t':int(t),'miou':{key:summarize(mats[key][j])['miou_4class'] for key in mats},'state_oracle_mse':drift[j],'clipping_changes_fraction':changed[j]} for j,t in enumerate(grid)]}
   results['trajectories'].append(row)
   print('TRAJECTORY',checkpoint,seed,json.dumps(row),flush=True)
  del model;torch.cuda.empty_cache()
 state=torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False)
 torch.cuda.manual_seed_all(42);ts=[]
 for _ in range(600):
  ts.extend(torch.randint(1000,(2,),device='cuda').cpu().tolist())
  torch.randn((2,4,256,256),device='cuda')
 matched=torch.equal(torch.cuda.get_rng_state(),state['rng']['cuda'][0])
 results['training_timestep_replay']={'final_cuda_rng_matches_checkpoint':matched,'sample_count':len(ts),'counts':{str(v):sum(t>=v for t in ts) for v in [900,950,990,999]},'best_epoch_counts':{str(v):sum(t>=v for t in ts[:1000]) for v in [900,950,990,999]}}
(out/'trajectory_diagnosis.json').write_text(json.dumps(results,indent=2));print('REPLAY',results['training_timestep_replay'],flush=True)

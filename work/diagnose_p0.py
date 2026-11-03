import os,sys,json,csv,time
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.data import FixedCropDataset,CLASSES
from conditional_diffusion.model import make_model
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.inference import evaluate,image_seed
from conditional_diffusion.metrics_impl import confusion,summarize
from conditional_diffusion.objectives import losses
from conditional_diffusion.runtime import seed_all
run=ROOT/'outputs/P0/diffusion/seed_42/diffusion_resume_20261017'
out=ROOT/'outputs/launches/diffusion_resume_20261017/diagnostics';out.mkdir(exist_ok=True)
torch.set_num_threads(4);seed_all(42)
results={'sampling':[],'teacher_forced':[],'loss_history':{}}
def save(): (out/'p0_diagnosis.json').write_text(json.dumps(results,indent=2))
for checkpoint in ['best','last']:
 state=torch.load(run/f'checkpoints/{checkpoint}.pt',map_location='cpu',weights_only=False)
 c=state['config'];ds=FixedCropDataset(c['manifest'],c['fixed_crops'])
 model=make_model(c).cuda().eval();model.load_state_dict(state['model'])
 diffusion=GaussianDiffusion(c['timesteps'],'cuda')
 print('CHECKPOINT',checkpoint,'epoch',state['epoch'],flush=True)
 for steps in ([1,5,20,50,100] if checkpoint=='best' else [1,20,100]):
  for seed in [3407,3408]:
   cfg=dict(c,sampling_steps=steps,sampling_seed=seed,preview_count=0)
   metrics,matrices,_=evaluate(model,diffusion,ds,cfg)
   row={'checkpoint':checkpoint,'epoch':state['epoch'],'steps':steps,'seed':seed,'metrics':metrics,
    'per_crop':[{ 'id':str(ident),**summarize(mat)} for ident,mat in zip(matrices['ids'],matrices['per_image'])]}
   results['sampling'].append(row);save()
   print('SAMPLING',checkpoint,steps,seed,round(metrics['miou_4class'],5),flush=True)
 with torch.no_grad():
  for tval in [0,100,500,800,900,950,990,999]:
   for condition in ['rgb','zero_rgb']:
    matrix=np.zeros((4,4),dtype=np.int64);sums={};amplitudes=[]
    for i in range(len(ds)):
     im,mask,ident=ds[i];im=im[None].cuda();mask=mask[None].cuda()
     if condition=='zero_rgb':im=torch.zeros_like(im)
     x0=diffusion.encode(mask);t=torch.tensor([tval],device='cuda')
     gen=torch.Generator(device='cuda').manual_seed(10000+i)
     noise=torch.randn(x0.shape,device='cuda',generator=gen)
     xt=diffusion.q_sample(x0,t,noise);raw=model(im,xt,t)
     _,terms=losses(raw,mask,c,x0)
     for k,v in terms.items():sums[k]=sums.get(k,0)+float(v)/len(ds)
     matrix+=confusion(mask[0].cpu().numpy(),raw.argmax(1)[0].cpu().numpy())
     amplitudes.append(float((raw.argmax(1)!=raw.clamp(-1,1).argmax(1)).float().mean()))
    row={'checkpoint':checkpoint,'t':tval,'condition':condition,'metrics':summarize(matrix),'losses':sums,
     'clipping_changes_argmax_fraction':sum(amplitudes)/len(amplitudes)}
    results['teacher_forced'].append(row);save()
    print('TEACHER',checkpoint,tval,condition,round(row['metrics']['miou_4class'],5),flush=True)
 del model;torch.cuda.empty_cache()
rows=list(csv.DictReader((run/'metrics_epochs.csv').open())) if (run/'metrics_epochs.csv').exists() else []
for tag in sorted(set(r['tag'] for r in rows)):
 if 'loss_' in tag or 'lr_' in tag:
  selected=[r for r in rows if r['tag']==tag]
  results['loss_history'][tag]=[{'epoch':int(r['epoch']),'value':float(r['value'])} for r in selected if int(r['epoch']) in [1,50,100,150,200,250,300]]
save(); print('DONE',str(out/'p0_diagnosis.json'),flush=True)

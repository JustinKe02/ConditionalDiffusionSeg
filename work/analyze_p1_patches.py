import os,sys,json
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np,torch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.data import MaterialDataset
from conditional_diffusion.model import make_model
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.inference import predict_image,image_seed
from conditional_diffusion.metrics_impl import confusion,summarize
from conditional_diffusion.runtime import seed_all
runroot=ROOT/'outputs/P1';batch='p1_cosine_20261017';out=ROOT/'outputs/launches'/batch/'analysis';out.mkdir(exist_ok=True)
c=json.loads((runroot/'diffusion/seed_42'/batch/'config.json').read_text())
torch.set_num_threads(4);seed_all(42)
ds=MaterialDataset(c['manifest'],'train',crop=512,seed=42,augment=True);ds.epoch=40
indices=np.random.default_rng(42).permutation(len(ds))[:12].tolist()
patches=[ds[i] for i in indices]
results={'scope':'Forward-only P1 best checkpoints; 12 prespecified random training crops at epoch40/seed42, actual 512 resolution and training augmentation. Not full-train evaluation, not val metrics.',
 'indices':indices,'ids':[x[2] for x in patches],'methods':{},'teacher_forced':{}}
npix=np.bincount(np.concatenate([x[1].numpy().ravel() for x in patches]),minlength=4)
results['crop_class_counts']=npix.tolist();results['crop_class_fractions']=(npix/npix.sum()).tolist()
for method in ['direct','iterative','diffusion']:
 state=torch.load(runroot/method/'seed_42'/batch/'checkpoints/best.pt',map_location='cpu',weights_only=False)
 cfg=state['config'];model=make_model(cfg).cuda().eval();model.load_state_dict(state['model']);d=GaussianDiffusion(1000,'cuda')
 matrix=np.zeros((4,4),dtype=np.int64)
 for image,target,ident in patches:
  pred,_=predict_image(model,d,image,ident,cfg);matrix+=confusion(target.numpy(),pred)
 results['methods'][method]=dict(checkpoint_epoch=state['epoch'],metrics=summarize(matrix),confusion=matrix.tolist())
 print('TRAIN_PATCHES',method,json.dumps(results['methods'][method]),flush=True)
 if method=='diffusion':
  with torch.no_grad():
   for tval in [0,500,950,999]:
    matrix=np.zeros((4,4),dtype=np.int64)
    for image,target,ident in patches:
     generator=torch.Generator().manual_seed(image_seed(ident,3407))
     noise=torch.randn((1,4,512,512),generator=generator).cuda()
     im=image[None].cuda();mask=target[None].cuda();t=torch.tensor([tval],device='cuda')
     xt=d.q_sample(d.encode(mask),t,noise);pred=model(im,xt,t).argmax(1)[0].cpu().numpy()
     matrix+=confusion(target.numpy(),pred)
    results['teacher_forced'][str(tval)]=summarize(matrix)
    print('TEACHER_FORCED',tval,json.dumps(results['teacher_forced'][str(tval)]),flush=True)
 del model,state;torch.cuda.empty_cache()
(out/'train_patch_diagnosis.json').write_text(json.dumps(results,indent=2));print('DONE',flush=True)

"""Train-split, image-only reference selection; labels enter only after candidate selection."""
import os,sys,hashlib,json
from pathlib import Path
import torch,numpy as np
from scipy import ndimage as ndi
from PIL import Image
from conditional_diffusion.data import MaterialDataset
from conditional_diffusion.diffusion_seg_experiment import pad_full_image,normalize,build
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
out=Path(sys.argv[1]); out.mkdir(exist_ok=True)
run=Path('outputs/diffusion_seg_20261103/direct')
with open(run/'last.pt','rb') as f:
 hsh=hashlib.sha256()
 while (b:=f.read(1<<20)):hsh.update(b)
 f.seek(0); state=torch.load(f,map_location='cpu',weights_only=False)
ep,step=state['epoch'],state['global_step']
cfg=dict(state['config'],device='cpu'); train=MaterialDataset(cfg['manifest'],'train',augment=False)
model,_=build(cfg);model.load_state_dict(state['model']);model.eval();del state
spec=dict(epoch=ep,step=step,sha256=hsh.hexdigest(),rule='at most four 32x32 anchors per thin class, one per 8-connected predicted majority component, select maximal minimum of 2x2 prior cell probabilities; ties row-major; complete native cells only',levels=[.70,.85,.95])
print(spec,flush=True)
images=[]
with torch.inference_mode():
 for it,ident in enumerate(train.ids):
  rgb,mask,_=train.raw(it); H,W=mask.shape
  arr,_=pad_full_image(rgb,mask,512)
  thumb=Image.fromarray(arr).resize((arr.shape[1]//4,arr.shape[0]//4),Image.Resampling.BOX)
  _,p,_=model.encode_global([normalize(np.asarray(thumb))])[0]
  p=p.softmax(0).numpy()[:,:H//16,:W//16]
  pred=p.argmax(0); candidates=[]
  for label in [1,2]:
   labs,_=ndi.label(pred==label,structure=np.ones((3,3)))
   # A 2x2 block entirely above 0.7 in this class is necessarily in one component.
   v=np.lib.stride_tricks.sliding_window_view(p[label],(2,2)).min(axis=(-2,-1));y,x=np.where(v>=.70)
   seq=sorted(zip(y,x),key=lambda z:-float(v[z]))
   seen=set()
   for yy,xx in seq:
    component=int(labs[yy,xx])
    if component in seen:continue
    seen.add(component); candidates.append(dict(label=label,y=int(yy*16),x=int(xx*16),score=float(v[yy,xx])))
    if len(seen)==4:break
  for label in [1,2]:
   actual,_=ndi.label(mask==label,structure=np.ones((3,3)))
   for a in [v for v in candidates if v['label']==label]:
    y,x=a['y'],a['x'];counts=np.bincount(actual[y:y+32,x:x+32].ravel());hit=counts[1:]
    a['purity']=float(hit.sum()/1024);a['gt_component']=int(np.argmax(hit)+1) if len(hit) else None
  images.append(dict(id=ident,gt_classes=[c for c in [1,2] if np.any(mask==c)],anchors=candidates))
  if (it+1)%25==0:print('completed',it+1,flush=True)
(out/'reference_check.json').write_text(json.dumps(dict(spec=spec,images=images),indent=2))
for thr in spec['levels']:
 d=dict(n=0,correct=0,any_img=0,two_true_regions=0,both_correct=0,any_good=0,byclass={i:[0,0] for i in (1,2)},thinimages=sum(bool(im['gt_classes']) for im in images),gtboth=sum(len(im['gt_classes'])==2 for im in images))
 for im in images:
  a=[a for a in im['anchors'] if a['score']>=thr]; g=[a for a in a if a['purity']>=.9]
  d['n']+=len(a); d['correct']+=len(g);d['any_img']+=bool(a);d['any_good']+=bool(g);d['two_true_regions']+=len(set((v['label'],v['gt_component']) for v in g))>=2
  d['both_correct']+=len(set(v['label'] for v in g))==2
  for v in a:
   d['byclass'][v['label']][0]+=1;d['byclass'][v['label']][1]+=v['purity']>=.9
 print(thr,d)

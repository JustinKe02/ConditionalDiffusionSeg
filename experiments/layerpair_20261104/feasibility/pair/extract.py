import json,cv2,numpy as np
from pathlib import Path
from PIL import Image
cv2.setNumThreads(1)
project=Path('/root/autodl-tmp/ConditionalDiffusionSeg')
m=json.load(open(project/'data_manifests/source_dataset.json')); f=Path(m['splits']['train']['snapshot_path'])
import hashlib
assert hashlib.sha256(f.read_bytes()).hexdigest()==m['splits']['train']['sha256']
out={}
for ident in f.read_text().split():
 root=Path(m['dataset_root'])
 mask=np.array(Image.open(root/m['mask_pattern'].format(id=ident)))
 img=np.asarray(Image.open(root/m['image_pattern'].format(id=ident)).convert('RGB'))
 regs=[]
 for c in (1,2):
  binary=(mask==c).astype('uint8')
  n,label,stats,ct=cv2.connectedComponentsWithStats(binary,8)
  for jj in np.argsort(-stats[1:,-1],kind='stable')[:6]:
   j=jj+1;x,y,w,h,a=stats[j]
   if a<256:continue
   inside=cv2.erode((label[y:y+h,x:x+w]==j).astype('uint8'),np.ones((5,5),'uint8'),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
   if inside.sum()<64:continue
   rgb=img[y:y+h,x:x+w][inside]
   regs.append({'c':c,'rgb':np.median(rgb,axis=0).tolist(),'area':int(a),'x':float(ct[j,0]),'y':float(ct[j,1])})
 out[ident]=regs
json.dump(out,open('regions.json','w'))
print('images, total comps, two classes, >=2 comps, >=2 each',len(out),sum(map(len,out.values())),sum(len(set(r['c'] for r in v))==2 for v in out.values()),sum(len(v)>=2 for v in out.values()),sum(all(sum(r['c']==c for r in v)>=2 for c in [1,2]) for v in out.values()),flush=True)

import json,sys,hashlib,time
from pathlib import Path
from PIL import Image
import cv2,numpy as np
cv2.setNumThreads(1)
p=Path(sys.argv[1]); ref=p/'reference_check.json'
while not ref.exists(): time.sleep(2)
an={r['id']:r['anchors'] for r in json.loads(ref.read_text())['images']}
m=json.load(open('data_manifests/source_dataset.json')); split=m['splits']['train']; raw=Path(split['snapshot_path']).read_bytes()
assert hashlib.sha256(raw).hexdigest()==split['sha256']
out={}
for ident in raw.decode().split():
 if ident=='m169':continue
 def load(k):return np.asarray(Image.open(Path(m['dataset_root'])/m[k+'_pattern'].format(id=ident)).convert('RGB' if k=='image' else 'L')) if k=='image' else np.asarray(Image.open(Path(m['dataset_root'])/m[k+'_pattern'].format(id=ident)))
 rgb,gt=load('image'),load('mask')
 a=[dict(x=a['x'],y=a['y'],cls=a['label'],score=a['score'],col=np.median(rgb[a['y']:a['y']+32,a['x']:a['x']+32],axis=(0,1)).tolist()) for a in an[ident] if a['score']>=.85]
 q=[]
 for cls in [1,2]:
  mask=np.uint8(gt==cls);n,cc,st,cen=cv2.connectedComponentsWithStats(mask,connectivity=8); core=cv2.erode(mask,np.ones((17,17),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
  for ix in range(1,n):
   x,y,w,h,pix=st[ix];local=(cc[y:y+h,x:x+w]==ix)&(core[y:y+h,x:x+w]>0)
   iy,ixx=np.nonzero(local)
   if not len(iy):continue
   center=((iy-iy.mean())**2+(ixx-ixx.mean())**2).argmin()
   cy,cx=int(iy[center])+y,int(ixx[center])+x
   feat=np.median(rgb[cy-8:cy+9,cx-8:cx+9],axis=(0,1)).tolist()
   ok=[k for k,b in enumerate(a) if not np.any(cc[b['y']:b['y']+32,b['x']:b['x']+32]==ix)]
   q.append(dict(cls=cls,col=feat,ar=int(pix),ok=ok))
 out[ident]={'a':a,'q':q}
(p/'joint_data.json').write_text(json.dumps(out))

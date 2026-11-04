import hashlib,json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
cv2.setNumThreads(1)
manifest=json.load(open("data_manifests/source_dataset.json"))
split=manifest["splits"]["train"]
raw=Path(split["snapshot_path"]).read_bytes()
assert hashlib.sha256(raw).hexdigest()==split["sha256"]
ids=raw.decode().split();root=Path(manifest["dataset_root"])
def file(kind,i):
 return root/manifest[kind+"_pattern"].format(id=i)
out={}
for i in ids:
 im=np.array(Image.open(file("mask",i)))
 row={}
 for cls in (1,2):
  binary=np.uint8(im==cls)
  n,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
  area=stats[1:,cv2.CC_STAT_AREA]
  core={}
  for nside in (9,17,33):
   inside=cv2.erode(binary,np.ones((nside,nside),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)!=0
   core[nside]=np.bincount(labels[inside],minlength=len(area)+1)[1:]
  row[cls]=[[int(k),int(a),int(b),int(c)] for k,a,b,c in zip(area,core[9],core[17],core[33])]
 out[i]=row
dup=[]
for i in ("m56","m169"):
 rgb=np.asarray(Image.open(file("image",i)).convert("RGB"))
 dup.append((i,hashlib.sha256(rgb.tobytes()).hexdigest()))
print(json.dumps(dict(train_split_sha256=split["sha256"],dup=dup,stats=out)))

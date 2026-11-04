"""Balanced within-image target classification among images with at least two distinct components per thin class."""
import json, numpy as np, collections
from scipy.special import expit,softmax
from scipy.optimize import minimize
regs=json.load(open('regions.json'));regs.pop('m169');fold=json.load(open('results.json'))['folds'];out={}
def ref(k):return all(sum(r['c']==z for r in regs[k])>=2 for z in (1,2))
eligible=[k for k in fold if ref(k)]
def go(ids,n,mode,rng,donors=None):
 donors=ids if donors is None else donors
 tag=[];listx=[];labels=[];w=[]
 for k in ids:
  if not ref(k):continue
  v=regs[k]
  for i,t in enumerate(v):
   c=t['c']; own=np.array(t['rgb'])/255.
   same=[[np.array(a['rgb'])/255. for j,a in enumerate(v) if j!=i and a['c']==z] for z in (1,2)]
   across=[[np.array(a['rgb'])/255. for key in donors if key!=k for a in regs[key] if a['c']==z] for z in (1,2)]
   for j in range(n):
    bank=across if mode=='cross' else same
    if mode=='target': x=own
    else:
     a,b=[u[rng.integers(len(u))] for u in bank]
     x=np.r_[own,own-a,own-b,np.abs(own-a),np.abs(own-b)]
    tag.append(k);listx.append(x);labels.append(c-1);w.append(1/(sum(r['c']==c for r in v)*2*n))
 return np.array(listx),np.array(labels),np.array(w),np.array(tag)
def fit(x,y,w):
 w=w/w.sum(); mu=w@x; sd=np.maximum(np.sqrt(w@(x-mu)**2),1e-4);z=np.c_[(x-mu)/sd,np.ones(len(x))]
 def obj(co):
  v=z@co;return (w@(np.logaddexp(0,v)-y*v)+0.005*np.sum(co[:-1]**2), z.T@(w*(expit(v)-y))+0.01*np.r_[co[:-1],0])
 co=minimize(obj,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B').x
 return lambda x:(x-mu)/sd@co[:-1]+co[-1]
by=collections.defaultdict(dict)
for f in range(5):
 tr=[k for k in fold if fold[k]!=f];te=[k for k in fold if fold[k]==f]
 if not any(map(ref,te)):continue
 for fitmode,valmode in [('same','same'),('same','cross'),('cross','cross'),('target','target')]:
  x,y,w,_=go(tr,32,fitmode,np.random.default_rng(130+f));mt=fit(x,y,w)
  xt,yt,wt,kt=go(te,64,valmode,np.random.default_rng(21+f));acc=(mt(xt)>0)==yt
  for k in set(kt):
   by[k][fitmode+'_'+valmode]=np.average(acc[kt==k],weights=wt[kt==k])
print('nImages',len(by),'targets',sum(len(regs[k]) for k in by))
a=np.array([[v[c] for c in ('same_same','same_cross','cross_cross','target_target')] for v in by.values()]);print('mean',a.mean(0)); res=a[np.random.default_rng(42).integers(len(a),size=(10000,len(a)))].mean(1);print('same-cross',np.percentile(res[:,0]-res[:,1],[2.5,97.5]),'same-target',np.percentile(res[:,0]-res[:,3],[2.5,97.5]))
json.dump({'note':'oracle anchors per class, preliminary; 16 joint-eligible images; fitted only on eligible train-fold images','n':len(by),'targetCount':sum(len(regs[k]) for k in by),'perImage':by},open('target_results.json','w'),indent=1)

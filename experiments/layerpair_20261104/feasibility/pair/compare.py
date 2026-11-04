"""Ordinal relative-colour probe. No fit stage uses val or test images."""
import numpy as np,json,collections,itertools
from pathlib import Path
from scipy.special import logsumexp,softmax
from scipy.optimize import minimize
regs=json.load(open('regions.json'))
# remove second copy entirely: stronger than merely assigning twins to same fold
regs.pop('m169')
rg=np.random.default_rng(213)
strata=collections.defaultdict(list)
for k,v in sorted(regs.items()):
 a,b=[sum(r['c']==c for r in v) for c in [1,2]]
 strata[(min(a,2),min(b,2))].append(k)
fold={}
for ids in strata.values():
 rg.shuffle(ids)
 start=int(rg.integers(5))
 for j,k in enumerate(ids):fold[k]=(j+start)%5
def feat(x,a,mode):
 x=np.asarray(x)/255.;a=np.asarray(a)/255.
 if mode=='target':return x
 d=x-a
 return np.r_[d,np.abs(d)]
def gen(ids,n,seed,mode):
 g=np.random.default_rng(seed); pools={c:[(k,r) for k in ids for r in regs[k] if r['c']==c] for c in (1,2)}
 crossbank={(k,c):[r for key,r in pools[c] if key!=k] for k in ids for c in (1,2)}
 tag=[];same=[];cross=[]
 for k in ids:
  v=regs[k]
  for i,j in itertools.permutations(range(len(v)),2):
   t,a=v[i],v[j];cls=int(np.sign(t['c']-a['c']))+1
   cand=crossbank[k,a['c']]
   if not cand: continue
   rr=g.integers(len(cand),size=n)
   tag.append([k,int(i),int(j),cls]);same.append(feat(t['rgb'],a['rgb'],mode));cross.append([feat(t['rgb'],cand[ii]['rgb'],mode) for ii in rr])
 return tag,np.stack(same),np.array(cross)
def fit(x,y,w):
 if x.ndim==3:
  npt,nr,nft=x.shape;x=x.reshape((-1,nft));y=np.repeat(y,nr);w=np.repeat(w/nr,nr)
 cen=np.average(x,axis=0,weights=w);scale=np.maximum(np.sqrt(np.average((x-cen)**2,axis=0,weights=w)),1e-5);scaled=(x-cen)/scale
 z=np.column_stack((scaled,np.ones(len(scaled))));hc=np.eye(3)[y]
 def fn(wts):
  wts=wts.reshape(-1,3);a=z@wts;q=softmax(a,axis=1)
  loss=np.sum(w*(logsumexp(a,axis=1)-np.sum(a*hc,axis=1)))+0.005*np.sum(wts[:-1]**2)
  grad=z.T@(w[:,None]*(q-hc));grad[:-1]+=0.01*wts[:-1];return loss,grad.ravel()
 weights=minimize(fn,np.zeros(z.shape[-1]*3),jac=True,method='L-BFGS-B').x.reshape(-1,3)
 return lambda X: softmax(((X-cen)/scale)@weights[:-1]+weights[-1],axis=-1)
def groupweights(tag):
 count=collections.Counter((a[0],a[-1]) for a in tag);cn=collections.Counter(s for k,s in count)
 return np.array([1/(3*cn[a[-1]]*count[a[0],a[-1]]) for a in tag])
rows=[]
for f in range(5):
 train=[k for k in fold if fold[k]!=f]; test=[k for k in fold if fold[k]==f]
 t,sa,ca=gen(train,5,300+f,'pair');te,s,c=gen(test,32,1300+f,'pair');lab=np.array([r[-1] for r in t]);true=np.array([r[-1] for r in te])
 wt=groupweights(t);model=fit(sa,lab,wt);m_cross=fit(ca,lab,wt)
 _,xs,_=gen(train,1,300+f,'target');_,xt,_=gen(test,1,1300+f,'target');base=fit(xs,lab,wt)
 rates=np.array([(model(s).argmax(-1)==true).astype(float),(model(c).argmax(-1)==true[:,None]).mean(-1),(m_cross(c).argmax(-1)==true[:,None]).mean(-1),(base(xt).argmax(-1)==true).astype(float)])
 # frozen contrast, trained-on-cross contrast, target-only. Average correctness over cross draws, not voting.
 for j,(k,qi,ai,y) in enumerate(te):rows.append([k,qi,ai,y,*rates[:,j].tolist()])
by=collections.defaultdict(lambda:collections.defaultdict(list))
for k,_,__,y,*v in rows:by[k][y].append(v)
out_by={k:{int(y):np.mean(v,axis=0).tolist() for y,v in cell.items()} for k,cell in by.items()}
keys=list(by)
arr=np.array([[out_by[k].get(y,[np.nan]*4) for y in range(3)] for k in keys])
def stat(v):return np.nanmean(np.nanmean(v,axis=0),axis=0)
res={}
for name, kk in [('all',keys),('both',[k for k in keys if len({r['c'] for r in regs[k]})==2])]:
 idx=[keys.index(k) for k in kk];v=arr[idx];point=stat(v);p=np.random.default_rng(444).integers(len(idx),size=(10000,len(idx)));bs=np.nanmean(np.nanmean(v[p],axis=1),axis=1)
 res[name]={'nImages':len(idx),'nOrderedPairs':sum(r[0] in kk for r in rows),'perRelationImages':np.sum(np.isfinite(v[:,:,0]),axis=0).tolist(),'mean':point.tolist(),'perRelation':np.nanmean(v,axis=0).tolist(),'ciSameMinusShuffle':np.percentile(bs[:,0]-bs[:,1],[2.5,97.5]).tolist(),'ciSameMinusSeparateCross':np.percentile(bs[:,0]-bs[:,2],[2.5,97.5]).tolist(),'ciSameMinusTarget':np.percentile(bs[:,0]-bs[:,3],[2.5,97.5]).tolist()}
print(json.dumps(res,indent=2))
json.dump({'columns':['sameRGBDelta','sameModelCrossRefs32','crossTrainedCrossRefs32','targetOnly'],'orderedRel':['thinner','same','thicker'],'results':res,'folds':fold,'perImageByRelation':out_by},open('results.json','w'),indent=1)

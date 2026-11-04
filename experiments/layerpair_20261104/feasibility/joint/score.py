from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
def fit_predictor(x,y,n_queries):
 x=np.array(x);y=np.array(y);mu=x.mean(0);sd=x.std(0);sd[sd<1e-9]=1
 X=np.c_[np.ones(len(y)),(x-mu)/sd]
 w=n_queries/(2*np.bincount(y)[y])
 def fg(b):
  z=X@b
  return (np.sum(w*(np.logaddexp(0,z)-y*z))+.5*np.sum(b[1:]**2),X.T@(w*(expit(z)-y))+np.r_[0,b[1:]])
 b=minimize(fg,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B').x
 return lambda v:expit(b[0]+((np.array(v)-mu)/sd)@b[1:])
def balanced_accuracy_score(y,pr):return np.mean([np.mean(pr[y==l]==l) for l in (0,1)])
def roc_auc_score(y,pr):
 n=y.sum();return (rankdata(pr)[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n))
p=Path(sys.argv[1]);d=json.loads((p/'joint_data.json').read_text())
rows=[(i,q,max([d[i]['a'][x] for x in q['ok']],key=lambda b:b['score'])) for i,r in d.items() for q in r['q'] if q['ok']]
print('Supported',len(rows),'of',sum(len(r['q']) for r in d.values()),'images',len({r[0] for r in rows}))
group=np.array([i for i,_,_ in rows]); y=np.array([q['cls']-1 for _,q,a in rows]);pred={k:np.zeros(len(rows)) for k in ('query','anchor_label_control','same','cross','swap')};rng=np.random.default_rng(27717)
def xfeat(row,anchor,kind):
 i,q,a=row;c=np.array(q['col'])
 if kind=='query':return c
 s=(-1 if anchor['cls']==1 else 1);core=np.r_[c,s]
 if kind=='anchor_label_control':return core
 delta=c-np.array(anchor['col'])
 return np.r_[core,delta,s*delta]
folds=[[] for _ in range(5)];sizes=[0]*5
for g in sorted(set(group),key=lambda g:(-sum(group==g),g)):
 k=np.argmin(sizes);folds[k].append(g);sizes[k]+=sum(group==g)
for fold in folds:
 te=np.where(np.isin(group,fold))[0];tr=np.where(~np.isin(group,fold))[0]
 bank=[(g,b) for g in sorted(set(group[tr])) for b in d[g]['a']]
 def samples(idx,kind):
  xx=[]; yy=[];idxmap=[]
  for h in idx:
   row=rows[h];i,q,a=row
   if kind=='cross':
    near=sorted((b for g,b in bank if g!=i and b['cls']==a['cls']),key=lambda b:abs(b['score']-a['score']))[:10]
   else:near=[a]
   for b in near:xx.append(xfeat(row,b,kind)); yy.append(y[h]);idxmap.append(h)
  return xx,yy,np.array(idxmap)
 for k in ('query','anchor_label_control','same','cross'):
  xx,yy,_=samples(tr,k); xt,_,imap=samples(te,k)
  mod=fit_predictor(xx,yy,len(tr));sc=mod(xt)
  pred[k][te]=[np.mean(sc[imap==u]) for u in te]
  if k=='same':
   xt,_,imap=samples(te,'cross'); sc=mod(xt);pred['swap'][te]=[np.mean(sc[imap==u]) for u in te]
scores={}
for k,v in pred.items():
 scores[k]=dict(bacc=balanced_accuracy_score(y,v>=.5),auc=roc_auc_score(y,v),fewrec=np.mean(v[y==1]>=.5),monorec=np.mean(v[y==0]<.5))
print('n1,n2',np.bincount(y),scores)
ids=np.unique(group);ns=np.array([[sum((group==g)&(y==c)) for c in [0,1]] for g in ids])
boot=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=5000)
for name in ['cross','swap','query','anchor_label_control']:
 diff=(np.equal(pred['same']>=.5,y)).astype(float)-np.equal(pred[name]>=.5,y)
 ds=np.array([[sum(diff[(group==g)&(y==c)]) for c in [0,1]] for g in ids])
 print('same minus',name, np.quantile(((boot@ds)/(boot@ns)).mean(axis=1),[.025,.975]))
(p/'joint_predictions.json').write_text(json.dumps([dict(id=i,cls=int(j+1),**{k:float(v[n]) for k,v in pred.items()}) for n,(i,j) in enumerate(zip(group,y))],indent=2))

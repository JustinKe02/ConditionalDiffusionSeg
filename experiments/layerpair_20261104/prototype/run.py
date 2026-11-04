"""Single frozen backbone shared by both heads. No validation-time selection."""
import argparse, csv, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from features import FrozenDirect
from heads import descriptors, predict_heads, train_heads, BATCH_SIZE, STEPS
from assessment import add_image, finalize

def write(path, obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def train_indices(im,ident):
    # Predicted class and conditional ambiguity stratification. No mask consulted.
    p=im['p']; mass=p[:,:,1]+p[:,:,2]
    js=np.flatnonzero(mass.ravel()>=.20);probs=p.reshape(-1,4)[js]
    ratio=probs[:,2]/(probs[:,1]+probs[:,2]); close=(ratio>.10)&(ratio<.90)
    group=2*(ratio>=.5).astype('u1')+close
    gen=np.random.default_rng(int.from_bytes(hashlib.sha256(('42:'+ident).encode()).digest()[:8],'little'))
    selected=[]
    for v in range(4):
        ids=js[group==v]
        selected.append(ids if len(ids)<=2048 else gen.choice(ids,2048,replace=False))
    return np.divmod(np.concatenate(selected),p.shape[1])

def main(repo,out):
    src=repo/'outputs/diffusion_seg_20261103'
    if json.loads((src/'status.json').read_text())['status']!='completed':
        raise RuntimeError('Direct has not finished its original queue')
    if shutil.disk_usage(out).free<9*1024**3:raise RuntimeError('Insufficient disk reserve')
    official=json.loads((src/'evaluations/direct.json').read_text())
    expected={row['id']:row for row in official['images']}
    ckpt=out/'frozen_direct_best.pt'
    if not ckpt.exists():shutil.copy2(src/'direct/best.pt',ckpt)
    if sha(ckpt)!=official['checkpoint_sha256']:raise RuntimeError('Backbone hash mismatch')
    torch.set_num_threads(4)
    mod=FrozenDirect(repo,ckpt,src/'configs/direct.json','cuda')
    if mod.epoch!=official['epoch'] or any(t.requires_grad for t in mod.model.parameters()):raise AssertionError('Unfrozen/mismatched weights')
    def status(phase,**kw):
        write(out/'status.json',dict(phase=phase,epoch=mod.epoch,**kw))
    prot=json.loads((src/'plan.json').read_text())
    write(out/'backbone.json',dict(epoch=mod.epoch,step=mod.global_step,sha256=sha(ckpt),expected_commit=prot['expected_commit'],config_sha=sha(src/'configs/direct.json'),splits=prot['provenance']['splits']))
    ds=mod.source('train'); chunks=[];train_counts=[]
    for i,ident in enumerate(ds.ids):
        if ident=='m169':continue
        rgb,mask,name=ds.raw(i);assert name==ident
        im=mod.image(rgb);ys,xs=train_indices(im,ident);c,a,b,e=descriptors(im,ys,xs)
        y=mask[ys,xs]; active=e&np.isin(y,[1,2]);ix=np.flatnonzero(active)
        chunks.append(dict(common=c[ix],selfref=a[ix],sameref=b[ix],p=im['p'][ys[ix],xs[ix]],y=y[ix],eligible=e[ix]))
        train_counts.append(dict(id=ident,selected=len(e),eligible=int(e.sum()),thin_labels=int(active.sum())))
        status('train_features',images=len(chunks),total=len(ds.ids)-1)
        del im,rgb,mask,c,a,b,e
    data={k:np.concatenate([v[k] for v in chunks]) for k in chunks[0]};del chunks
    write(out/'train_sampling.json',train_counts)
    status('fitting_heads');heads=train_heads(data,seed=42)
    dcounts=np.bincount(data['y'],minlength=4).tolist();del data
    nparams=[sum(p.numel() for p in head.parameters() if p.requires_grad) for head in heads]
    assert nparams[0]==nparams[1]
    torch.save(dict(A=heads[0].state_dict(),B=heads[1].state_dict(),backbone_sha=official['checkpoint_sha256']),out/'heads.pt')
    write(out/'budget.json',dict(parameters=nparams,backbone_trainable=0,updates_per_arm=STEPS,batch=BATCH_SIZE,pixel_draws_per_arm=STEPS*BATCH_SIZE,training_seed=42,pool_class_counts=dcounts))
    val=mod.source('val');assert set(val.ids)==set(expected) and len(val.ids)==25
    rows=[];audit=[]
    for i,ident in enumerate(val.ids):
        rgb,mask,name=val.raw(i);assert name==ident
        im=mod.image(rgb);p=im['p'];base=p.argmax(-1).astype('uint8');A=base.copy();B=base.copy()
        nactive=err_active=canflip=0
        js=np.flatnonzero((p[:,:,1]+p[:,:,2]).ravel()>=.20)
        for start in range(0,len(js),32768):
            ids=js[start:start+32768]; yy,xx=np.divmod(ids,p.shape[1])
            c,a,b,e=descriptors(im,yy,xx);prob=p[yy,xx]
            A[yy,xx],B[yy,xx]=predict_heads(heads,c,a,b,prob,e)
            nactive+=int(e.sum()); err=(mask[yy,xx]==2)&(base[yy,xx]==1)&e;err_active+=int(err.sum())
            v=prob[err]; diff=np.log(v[:,2].clip(1e-12))-np.log(v[:,1].clip(1e-12))+6.
            few=(v[:,1]+v[:,2])/(1.+np.exp(-diff));canflip+=int(((diff>0)&(few>np.maximum(v[:,0],v[:,3]))).sum())
        add_image(rows,ident,mask,base,A,B)
        obs=rows[-1]['direct']
        for key in ('confusion','boundaries'):
            if not np.array_equal(expected[ident][key],obs[key]):
                raise RuntimeError('Original validation counts not reproduced: '+ident+' '+key)
        audit.append(dict(id=ident,anchors=len(im['anchors']),pixels=int(mask.size),eligible=nactive,gt_few=int((mask==2).sum()),base_few_to_mono=int(((mask==2)&(base==1)).sum()),eligible_base_few_to_mono=err_active,reachable_under_cap=canflip))
        status('validation',images=i+1,total=len(val.ids))
        del im,rgb,mask,base,p,A,B
    finalize(rows,out);write(out/'eligibility.json',audit)
    status('completed',checked_direct_images=len(rows),finished_unix=time.time())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args(); args.output.mkdir(exist_ok=True,parents=True)
    try:main(args.repo,args.output)
    except Exception as e:
        write(args.output/'status.json',dict(phase='failed',message=str(e),traceback=traceback.format_exc()))
        raise

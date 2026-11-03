"""Meaningful CUDA equivalence, exact resume, control-isolation and reference checks."""
import sys, json, time, gc, subprocess, hashlib, inspect
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import core as C
G=C.G
OUT=C.BATCH/'preflight'

def same(a,b,path=''):
    if torch.is_tensor(a):
        assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a: same(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)): same(x,y,path+'/'+str(i))
    else: assert a==b,(path,a,b)

def compare_checkpoints(p,q):
    a=torch.load(p,map_location='cpu',weights_only=False); b=torch.load(q,map_location='cpu',weights_only=False)
    fields=['model','optimizer','scheduler','ema','rng','config','epoch','global_step','best_miou','plan_sha256']
    for f in fields: same(a[f],b[f],f)
    out=dict(fields=fields,epoch=a['epoch'],global_step=a['global_step'],sha256=[G.digest(p),G.digest(q)])
    del a,b; gc.collect()
    for path in (p,q):
        assert path.resolve().is_relative_to(OUT.resolve())
        path.unlink()  # engineering weights only, after equality and hashes recorded
    return out

def worker(kind,arm=None):
    OUT.mkdir(parents=True,exist_ok=True)
    if kind=='old_equiv':
        import glpflow_rollout_20261022 as legacy
        G.train(legacy.config(),OUT/'old_equiv','preflight',max_batches=2,stop_after_epoch=2)
    elif kind=='new_equiv': C.train(C.config('R'),OUT/'new_equiv','preflight',max_batches=2,stop_after_epoch=2)
    elif kind=='continuous': C.train(C.config(arm),OUT/f'{arm}_continuous','preflight',max_batches=2,stop_after_epoch=2)
    elif kind=='split1': C.train(C.config(arm),OUT/f'{arm}_resumed','preflight',max_batches=2,stop_after_epoch=1)
    elif kind=='split2': C.train(C.config(arm),OUT/f'{arm}_resumed','preflight',max_batches=2,stop_after_epoch=2)
    elif kind=='trace': trace_checks()
    elif kind=='reference': reference()
    elif kind=='smoke': smoke(arm)
    else: raise ValueError(kind)

def batch(c):
    ds=G.GLPDataset(c,'train'); it=iter(G.make_loader(ds,c,1)); b=next(it); del it
    return b

def trace_checks():
    b=batch(C.config('R')); traces={}; details={}
    for arm in ['R','I','N','F']:
        c=C.config(arm); model,opt,sched,ema=G.build(c)
        tr={}; terms,grad=C.train_step(model,opt,b,c,'cuda',tr)
        assert tr['endpoint_supervised_samples']==8 and bool((tr['level']==1).all())
        assert not tr['prefix_grad_enabled'] and tr['feature_grad_enabled']
        assert tr['local_grad_nonzero'] and tr['global_grad_nonzero']
        traces[arm]=tr; details[arm]={'terms':terms,'grad':grad,'supervised_samples':8,'level':1}
        del model,opt,sched,ema; gc.collect(); torch.cuda.empty_cache()
    r,i,n,f=[traces[k] for k in ['R','I','N','F']]
    for key in ['initial','eps','k','level','p','u','gtok','gpos','sc']: same(r[key],i[key],key)
    for key in ['eps','k','level','p','u','gtok','gpos']:
        same(r[key],n[key],key); same(n[key],f[key],key)
    k=i['k'].float()/4
    expected=(1-k)[:,None,None,None]*i['initial']+k[:,None,None,None]*G.encode_targets(b['mask'])
    assert torch.allclose(i['x'],expected,atol=1e-7,rtol=0)
    zero=i['k']==0
    if zero.any(): same(r['x'][zero],i['x'][zero],'I/R k0')
    same(n['initial'],2*n['p']-1,'N deterministic initial'); same(n['initial'],f['initial'],'N/F initial')
    for key in ['mask','gt','target','truth']: assert key not in inspect.signature(C.predict).parameters
    toy=torch.randn(4,4,5,5); prob=toy.softmax(1); t=torch.tensor([0.,.25,.5,.75]); dt=torch.full((4,),.25)
    same(C.advance(C.config('N'),toy,prob,t,dt),G.euler(toy,2*prob-1,t,dt),'Euler')
    same(C.advance(C.config('F'),toy,prob,t,dt),2*prob-1,'feedback')
    assert torch.allclose(C.advance(C.config('N'),toy,prob,t,dt)[3],(2*prob-1)[3],atol=1e-6)
    G.atomic_json(OUT/'trace_checks.json',dict(passed=True,actual_batch_shape=list(b['image'].shape),k=r['k'].tolist(),checks=details,
                  verified=['I/R identical actual conditioning except selected input state','I k0 and GT interpolation formula','noise draw retained','N/F update-only difference','8/8 endpoint supervision','level1','prefix detached','local/global feature gradients','predict no GT parameter']))

def smoke(arm):
    c=C.config(arm); b=batch(c); model,opt,sched,ema=G.build(c)
    torch.cuda.reset_peak_memory_stats(); start=None
    for j in range(5):
        if j==2: torch.cuda.synchronize(); start=time.perf_counter()
        terms,grad=C.train_step(model,opt,b,c,'cuda'); ema.update(model)
    torch.cuda.synchronize(); seconds=time.perf_counter()-start
    peak=torch.cuda.max_memory_reserved()/2**20
    assert peak<22*1024,(arm,peak)
    assert all(np.isfinite(v) for v in terms.values()) and np.isfinite(grad)
    G.atomic_json(OUT/f'smoke_{arm}.json',dict(passed=True,shape=list(b['image'].shape),warmup=2,timed_updates=3,samples_per_second=24/seconds,peak_reserved_mib=peak,terms=terms))

def reference():
    model,c,meta=C.load_best('R'); folder=C.ANALYSIS/'R'/'masks'; folder.mkdir(parents=True,exist_ok=True)
    metrics,mats,_=C.evaluate(model,c,on_image=lambda ident,p:Image.fromarray(p).save(folder/f'{ident}.png'))
    with np.load(C.RUNS['R']/'best_val_confusion.npz') as old:
        for k in ['ids','per_image','global_matrix','boundary_counts','per_image_boundary']: same(mats[k],old[k],k)
    np.savez_compressed(C.ANALYSIS/'R'/'best_val_confusion.npz',**mats)
    G.atomic_json(OUT/'reference_R.json',dict(passed=True,metrics=metrics,meta=meta,all25_counts_exact=True))
    d=G.GLPDataset(c,'val',train=False)[0]
    for arm in ['N','F']:
        cc=C.config(arm)
        p1,_=C.predict(model,cc,d['image'],d['thumb'],d['id'],sampling_seed=3407)
        p2,_=C.predict(model,cc,d['image'],d['thumb'],d['id'],sampling_seed=3408)
        same(p1,p2,arm+' deterministic source seed invariance')
    G.atomic_json(OUT/'noise_free_seed_invariance.json',dict(passed=True,id=d['id'],arms=['N','F'],scope='shared held-fixed R weights; structural source test; final N/F checkpoints checked again after training'))

def main():
    OUT.mkdir(parents=True,exist_ok=True); report={'started_unix':time.time(),'checks':{}}
    def call(kind,arm=None):
        args=[sys.executable,'-u','-B',__file__,'--worker',kind]+([] if arm is None else [arm])
        name=kind+('' if arm is None else '_'+arm)
        print('PREFLIGHT',name,flush=True)
        with (OUT/f'{name}.log').open('a') as log: subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    call('old_equiv'); call('new_equiv')
    report['checks']['old_new_R_full_state_exact']=compare_checkpoints(OUT/'old_equiv/checkpoints/last.pt',OUT/'new_equiv/checkpoints/last.pt')
    G.atomic_json(OUT/'progress.json',report)
    call('trace'); report['checks']['control_isolation']=json.loads((OUT/'trace_checks.json').read_text())
    for arm in ['I','N','F']:
        call('smoke',arm)
        for kind in ['continuous','split1','split2']: call(kind,arm)
        report['checks']['resume_'+arm]=compare_checkpoints(OUT/f'{arm}_continuous/checkpoints/last.pt',OUT/f'{arm}_resumed/checkpoints/last.pt')
        report['checks']['smoke_'+arm]=json.loads((OUT/f'smoke_{arm}.json').read_text())
        G.atomic_json(OUT/'progress.json',report)
    call('reference'); report['checks']['reference_R']=json.loads((OUT/'reference_R.json').read_text())
    report.update(passed=True,finished_unix=time.time())
    G.atomic_json(C.BATCH/'engineering_verification.json',report)
    print('PREFLIGHT PASSED',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker': worker(sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
    else: main()

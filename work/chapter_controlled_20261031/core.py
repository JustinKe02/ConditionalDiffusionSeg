"""Explicit controlled training/prediction; frozen legacy modules are read-only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, copy, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'work'))
import glpflow_core as G

BATCH = ROOT/'outputs/launches/chapter_controlled_ablation_20261031'
ANALYSIS = ROOT/'outputs/analysis/chapter_controlled_ablation_20261031'
PROPOSAL = ROOT/'outputs/plans/chapter_controlled_ablation_20261031/proposal.json'
OLD = ROOT/'outputs/launches/p1_glpflow_rollout_20261022/plan.json'
OLD_PLAN = json.loads(OLD.read_text())
RUNS = {a: ROOT/f'outputs/P1/glpflow/seed_42/chapter_controlled_20261031_{a.lower()}' for a in ['I','N','F']}
RUNS['R'] = Path(OLD_PLAN['run'])

def config(a):
    c = copy.deepcopy(OLD_PLAN['config'])
    if a == 'R': return c
    assert a in ['I','N','F']
    c.update(arm=a, controlled_arm=a, controlled_schema='20261031_v1',
             training_state='matched_gt_interpolation' if a=='I' else 'sampler_rollout_k4_no_grad',
             source_noise_scale=1. if a=='I' else 0.,
             state_update='endpoint_feedback' if a=='F' else 'euler')
    return c

def advance(c, x, q, t, dt):
    return 2*q-1 if c.get('controlled_arm')=='F' else G.euler(x, 2*q-1, t, dt)

def train_step(model, opt, batch, c, device, trace=None):
    image, mask, boxes = batch['image'].to(device), batch['mask'].to(device), batch['box']
    b, s = image.shape[0], image.shape[-1]; K = c['sampling_steps']
    feats = model.local(image)
    ctx = model.global_context([t.to(device) for t in batch['thumb']])
    p, umap, gtok, gpos, _ = G.window_conditions(model, c, ctx, boxes, s, device)
    prior_loss = image.new_zeros(())
    for (_, pl, _), soft in zip(ctx, batch['soft']):
        soft = soft.to(device)
        prior_loss = prior_loss+(-(soft*pl.log_softmax(0)).sum(0).mean()+G.dice_loss(pl[None], soft.argmax(0)[None]))/2
    prior_loss = prior_loss/b
    lv = torch.full((b,), G.level_for_steps(K), device=device, dtype=torch.long)
    k = torch.randint(K, (b,), device=device)
    eps = torch.randn(b, 4, s, s, device=device)
    x = G.make_source(c, p, umap, eps); sc = torch.zeros_like(p)
    initial = x.clone() if c.get('controlled_arm')=='I' else None
    if trace is not None:
        trace.update(initial=x.detach().cpu().clone(), eps=eps.cpu(), k=k.cpu(), level=lv.cpu(),
                     p=p.cpu(), u=umap.cpu(), gtok=gtok.detach().cpu(), gpos=gpos.cpu())
    prefix_samples=0; batch_calls=1
    with torch.no_grad():
        for j in range(K-1):
            idx = (k > j).nonzero().flatten()
            if not len(idx): break
            t = torch.full((len(idx),), j/K, device=device)
            out = model.denoise([f[idx] for f in feats], x[idx], p[idx], umap[idx], sc[idx], t, lv[idx],
                                gtok[idx], gpos[idx], [boxes[int(i)] for i in idx]).softmax(1)
            x[idx] = advance(c, x[idx], out, t, torch.full_like(t, 1/K)); sc[idx] = out
            prefix_samples += len(idx); batch_calls += 1
    t = k.float()/K
    if initial is not None:
        x = (1-t)[:,None,None,None]*initial+t[:,None,None,None]*G.encode_targets(mask)
    if trace is not None:
        trace.update(x=x.detach().cpu(), sc=sc.detach().cpu(), prefix_grad_enabled=x.requires_grad,
                     feature_grad_enabled=feats[0].requires_grad, endpoint_supervised_samples=b)
    logits = model.denoise(feats, x.detach(), p, umap, sc.detach(), t, lv, gtok, gpos, boxes)
    terms = dict(loss_ce=F.cross_entropy(logits, mask, reduction='none').mean(), loss_dice_fg=G.dice_loss(logits, mask), loss_prior=prior_loss)
    total = .5*terms['loss_ce']+.5*terms['loss_dice_fg']+c['prior_loss_weight']*prior_loss
    terms['loss_total'] = total
    if not torch.isfinite(total): raise FloatingPointError('Nonfinite loss')
    opt.zero_grad(set_to_none=True); total.backward()
    grad = nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'], error_if_nonfinite=True); opt.step()
    terms['mean_k'] = k.float().mean()
    result = {n: float(v.detach()) for n,v in terms.items()}
    result.update(endpoint_forward_samples=b+prefix_samples, endpoint_batch_calls=batch_calls)
    if trace is not None:
        trace['local_grad_nonzero'] = any(p.grad is not None and bool((p.grad!=0).any()) for p in model.local.parameters())
        trace['global_grad_nonzero'] = any(p.grad is not None and bool((p.grad!=0).any()) for p in model.global_stream.parameters())
    return result, float(grad)

@torch.no_grad()
def infer_windows(model,c,img,boxes,ctx,eps,stop_after=4,intervention=None,capture=()):
    assert stop_after in [1,2,4]
    b,sz=img.shape[0],img.shape[-1]; device=img.device
    feats=model.local(img)
    p,u,gtok,gpos,_=G.window_conditions(model,c,ctx,boxes,sz,device)
    x=G.make_source(c,p,u,eps); sc=torch.zeros_like(p)
    lv=torch.full((b,),1,device=device,dtype=torch.long)
    endpoints={}
    for j in range(stop_after):
        t=torch.full((b,),j/4,device=device)
        if j==1 and intervention=='zero_state': x=torch.zeros_like(x)
        if j==1 and intervention=='zero_selfcond': sc=torch.zeros_like(sc)
        logits=model.denoise(feats,x,p,u,sc,t,lv,gtok,gpos,boxes)
        sc=logits.softmax(1); x=advance(c,x,sc,t,torch.full_like(t,.25))
        if j+1 in capture: endpoints[j+1]=logits
    return logits,b*stop_after,endpoints

@torch.no_grad()
def predict(model,c,image,thumb,ident,stop_after=4,sampling_seed=None,return_prob=False,intervention=None,capture=(),timing=None):
    device=next(model.parameters()).device
    seed=c['sampling_seed'] if sampling_seed is None else sampling_seed
    h,w=image.shape[-2:]; sz=c['crop_size']; hw=((h//64)*64,(w//64)*64)
    if timing is not None: torch.cuda.synchronize(); t0=time.perf_counter()
    ctx=model.global_context([thumb.to(device)])
    if timing is not None: torch.cuda.synchronize(); timing['network_seconds']=time.perf_counter()-t0
    gen=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'{ident}:{seed}'.encode()).digest()[:8],'little')%(2**63-1))
    field=torch.randn((4,h,w),generator=gen)
    coords=[(y,x) for y in G.positions(h,sz,c['stride']) for x in G.positions(w,sz,c['stride'])]
    requested=sorted(set(capture)|{stop_after})
    total={k:torch.zeros((4,h,w)) for k in requested}; counts=torch.zeros((h,w)); nfe=0
    for start in range(0,len(coords),c['window_batch_size']):
        part=coords[start:start+c['window_batch_size']]; b=len(part)
        img=torch.stack([image[:,y:y+sz,x:x+sz] for y,x in part]).to(device)
        boxes=torch.tensor([[y/hw[0],x/hw[1],(y+sz)/hw[0],(x+sz)/hw[1]] for y,x in part])
        eps=torch.stack([field[:,y:y+sz,x:x+sz] for y,x in part]).to(device)
        if timing is not None: torch.cuda.synchronize(); t0=time.perf_counter()
        logits,calls,ends=infer_windows(model,c,img,boxes,ctx*b,eps,stop_after,intervention,requested)
        if timing is not None: torch.cuda.synchronize(); timing['network_seconds']+=time.perf_counter()-t0
        nfe+=calls
        for k in requested:
            prob=ends[k].softmax(1).cpu()
            for pr,(y,x0) in zip(prob,part): total[k][:,y:y+sz,x0:x0+sz]+=pr
        for y,x0 in part: counts[y:y+sz,x0:x0+sz]+=1
    assert bool((counts>0).all())
    probs={k:total[k]/counts for k in requested}
    preds={k:v.argmax(0).numpy().astype(np.uint8) for k,v in probs.items()}
    if capture: return preds,nfe
    return (preds[stop_after],nfe,probs[stop_after]) if return_prob else (preds[stop_after],nfe)

@torch.no_grad()
def evaluate(model,c,sampling_seed=None,on_image=None,limit=None,stop_after=4,intervention=None):
    model.eval(); ds=G.GLPDataset(c,'val',train=False)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    matrices=[]; bounds=[]; ids=[]; previews={}; nfe=0
    for i in range(len(ds) if limit is None else min(limit,len(ds))):
        d=ds[i]; pred,calls=predict(model,c,d['image'],d['thumb'],d['id'],stop_after,sampling_seed,intervention=intervention)
        truth=d['mask'].numpy(); nfe+=calls
        matrices.append(G.confusion(truth,pred)); bounds.append(G.boundary_counts(truth,pred,c['boundary_tolerance'])); ids.append(d['id'])
        if i<c['preview_count']: previews[d['id']]=G.preview(d['image'],truth,pred)
        if on_image: on_image(d['id'],pred)
    torch.cuda.synchronize(); m=np.sum(matrices,0); bc=np.sum(bounds,0)
    metrics={**G.classification_metrics(m),**G.summarize_boundaries(bc),'seconds':time.perf_counter()-start,
             'image_count':len(ids),'nfe_total':nfe,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
    return metrics,dict(ids=np.array(ids),per_image=np.array(matrices),global_matrix=m,boundary_counts=bc,per_image_boundary=np.array(bounds)),previews

def train(c,run,plan_sha,max_batches=None,stop_after_epoch=None,guard_fn=None):
    run=Path(run); device=c['device']; model,opt,sched,ema=G.build(c,device)
    last=run/'checkpoints/last.pt'; epoch0=step=0; best=-1.
    sg=G.StorageGuard(ROOT,run,c['storage'])
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        assert state['config']==c and state['plan_sha256']==plan_sha,'Resume config or plan changed'
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler'])
        ema.load_state_dict(state['ema']); G.restore_rng(state['rng'])
        epoch0,step,best=state['epoch'],state['global_step'],state['best_miou']; del state
    else:
        (run/'checkpoints').mkdir(parents=True,exist_ok=True); G.atomic_json(run/'config.json',c)
        G.atomic_json(run/'run_metadata.json',dict(parameters=sum(p.numel() for p in model.parameters()),arm=c['arm'],plan_sha256=plan_sha,torch=torch.__version__,gpu=torch.cuda.get_device_name()))
        G.atomic_checkpoint(last,G.checkpoint_payload(model,opt,sched,ema,c,0,0,best,plan_sha),sg)
    if epoch0>=c['epochs']: return dict(status='completed',epoch=epoch0)
    ds=G.GLPDataset(c,'train'); logger=G.ExperimentLogger(run,sg,step if epoch0 else None,epoch0 if epoch0 else None)
    G.atomic_json(run/'status.json',dict(status='running',last_completed_epoch=epoch0,global_step=step))
    try:
        for epoch in range(epoch0+1,c['epochs']+1):
            if guard_fn: guard_fn()
            model.train(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
            sums={}; seen=0; lr=opt.param_groups[0]['lr']; forward_samples=0; batch_calls=0
            for bi,batch in enumerate(G.make_loader(ds,c,epoch)):
                if max_batches is not None and bi>=max_batches: break
                terms,grad=train_step(model,opt,batch,c,device); ema.update(model)
                forward_samples+=terms['endpoint_forward_samples']; batch_calls+=terms['endpoint_batch_calls']
                n=batch['image'].shape[0]; seen+=n; step+=1
                for k,v in terms.items(): sums[k]=sums.get(k,0.)+v*n
                if step%c['log_every']==0:
                    logger.scalars('steps',{'train_step/'+k:v for k,v in dict(**terms,grad_norm=grad,lr=lr).items()},epoch,step)
                    G.atomic_json(run/'progress.json',dict(epoch=epoch,global_step=step,loss=terms['loss_total'],updated_unix=time.time()))
            torch.cuda.synchronize(); secs=time.perf_counter()-t0
            vals={'train_epoch/'+k:v/seen for k,v in sums.items()}
            vals.update({'resources/train_seconds':secs,'resources/samples_per_second':seen/secs,'resources/peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                         'compute/endpoint_forward_samples':forward_samples,'compute/endpoint_batch_calls':batch_calls,'optim/lr':lr})
            logger.scalars('epochs',vals,epoch,step); metrics=None
            if max_batches is None and (epoch%c['val_every']==0 or epoch==c['epochs']):
                metrics,mats,previews=evaluate(ema.shadow,c)
                logger.scalars('val',{'val/'+k:v for k,v in metrics.items()},epoch,step); logger.images(previews,epoch)
                G.atomic_json(run/f'validation_epoch{epoch}.json',metrics)
                if epoch==c['epochs']: np.savez_compressed(run/'epoch100_val_confusion.npz',**mats)
                if metrics['miou_4class']>best:
                    best=metrics['miou_4class']
                    G.atomic_checkpoint(run/'checkpoints/best.pt',dict(model=ema.shadow.state_dict(),config=c,epoch=epoch,global_step=step,
                                      validation_metrics=metrics,plan_sha256=plan_sha,weights='power_ema'),sg)
                    G.atomic_json(run/'best_val_metrics.json',dict(epoch=epoch,metrics=metrics)); np.savez_compressed(run/'best_val_confusion.npz',**mats)
            sched.step(); logger.flush()
            G.atomic_checkpoint(last,G.checkpoint_payload(model,opt,sched,ema,c,epoch,step,best,plan_sha),sg)
            G.atomic_json(run/'status.json',dict(status='completed' if epoch==c['epochs'] else 'running',last_completed_epoch=epoch,global_step=step,best_miou=best))
            print(dict(arm=c['arm'],epoch=epoch,step=step,loss=round(sums['loss_total']/seen,5),sec=round(secs,1),val=None if metrics is None else round(metrics['miou_4class'],5)),flush=True)
            if stop_after_epoch is not None and epoch>=stop_after_epoch: break
    except BaseException as e:
        G.atomic_json(run/'status.json',dict(status='interrupted',error=str(e),global_step=step)); raise
    finally: logger.close()
    return dict(status='completed' if epoch==c['epochs'] else 'stopped',epoch=epoch,best=best)

def load_best(a):
    G.configure(42); c=config(a)
    path=RUNS[a]/'checkpoints/best.pt'; state=torch.load(path,map_location='cpu',weights_only=False)
    model=G.GLPFlowNet(c).cuda(); model.load_state_dict(state['model']); model.eval()
    return model,c,dict(arm=a,checkpoint=str(path),checkpoint_sha256=G.digest(path),epoch=state['epoch'])

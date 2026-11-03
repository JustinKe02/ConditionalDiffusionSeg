"""Fixed val-only exports, boundary metrics, paired descriptive intervals and report."""
import sys,json,time,csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
import core as C
G=C.G

def boundary_counts(gt,pred,d):
    rows=[]; structure=np.ones((3,3),bool)
    for k in range(4):
        a,b=gt==k,pred==k
        # border_value1 has the same min/erosion result as replicate-edge padding.
        ag=a&~ndi.binary_erosion(a,structure=structure,border_value=1)
        bp=b&~ndi.binary_erosion(b,structure=structure,border_value=1)
        near_a=ndi.maximum_filter(ag,size=2*d+1,mode='constant',cval=0)
        near_b=ndi.maximum_filter(bp,size=2*d+1,mode='constant',cval=0)
        rows.append([np.count_nonzero(bp&near_a),np.count_nonzero(bp),np.count_nonzero(ag&near_b),np.count_nonzero(ag)])
    return np.asarray(rows,np.int64)

def boundary_iou_counts(gt,pred,d):
    rows=[]; structure=np.ones((3,3),bool)
    for k in range(4):
        a,b=gt==k,pred==k
        a=a&~ndi.binary_erosion(a,structure=structure,iterations=d,border_value=0)
        b=b&~ndi.binary_erosion(b,structure=structure,iterations=d,border_value=0)
        rows.append([np.count_nonzero(a&b),np.count_nonzero(a|b)])
    return np.asarray(rows,np.int64)

def boundary_iou_score(counts):
    vals=np.divide(counts[:,0],counts[:,1],out=np.full(4,np.nan),where=counts[:,1]>0)
    return dict(per_class=vals.tolist(),foreground=float(np.nanmean(vals[1:])))

def array_scores(cm,bc):
    tp=np.diagonal(cm,axis1=-2,axis2=-1); den=cm.sum(-1)+cm.sum(-2)-tp
    iou=np.divide(tp,den,out=np.full(tp.shape,np.nan),where=den>0)
    p=np.divide(bc[...,0],bc[...,1],out=np.zeros(bc.shape[:-1]),where=bc[...,1]>0)
    r=np.divide(bc[...,2],bc[...,3],out=np.zeros(bc.shape[:-1]),where=bc[...,3]>0)
    f=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r>0)
    f=np.where(bc[...,1]+bc[...,3]>0,f,np.nan)
    return {'miou':np.nanmean(iou,axis=-1),'boundary_f1':np.nanmean(f[...,1:],axis=-1),'thin_mean_iou':np.nanmean(iou[...,1:3],axis=-1)}

def metric_selfcheck():
    g=np.zeros((19,23),np.uint8); g[:7,:9]=1; g[8:17,9:20]=2; g[-3:,-4:]=3
    p=np.roll(g,1,axis=0)
    for d in [1,2,4]: assert np.array_equal(boundary_counts(g,p,d),G.boundary_counts(g,p,d))
    b=boundary_iou_counts(g,g,2); assert np.all(b[:,0]==b[:,1])
    z=np.zeros_like(g); assert boundary_iou_counts(z,z,1)[0,0]>0 # image border included

def arm(a):
    metric_selfcheck(); model,c,meta=C.load_best(a)
    out=C.ANALYSIS/a; out.mkdir(parents=True,exist_ok=True)
    ds=G.GLPDataset(c,'val',train=False); data={}; ids=[]; started=time.time()
    for step in [1,2,4]:
        data[f's{step}_cm']=[]
        for d in [1,2,4]: data[f's{step}_bf{d}']=[]; data[f's{step}_biou{d}']=[]
        (out/'endpoints'/str(step)).mkdir(parents=True,exist_ok=True)
    with np.load(C.RUNS[a]/'best_val_confusion.npz') as z: expected={k:z[k].copy() for k in z.files}
    assert np.array_equal(expected['ids'],np.array(ds.ids))
    for i in range(len(ds)):
        item=ds[i]; gt=item['mask'].numpy()
        preds,nfe=C.predict(model,c,item['image'],item['thumb'],item['id'],capture=(1,2,4))
        ids.append(item['id'])
        for step,pred in preds.items():
            Image.fromarray(pred).save(out/'endpoints'/str(step)/f"{item['id']}.png")
            cm=G.confusion(gt,pred); data[f's{step}_cm'].append(cm)
            for d in [1,2,4]:
                bc=boundary_counts(gt,pred,d); data[f's{step}_bf{d}'].append(bc)
                data[f's{step}_biou{d}'].append(boundary_iou_counts(gt,pred,d))
                if step==4 and d==2:
                    assert np.array_equal(bc,expected['per_image_boundary'][i]),(a,item['id'],'boundary')
            if step==4: assert np.array_equal(cm,expected['per_image'][i]),(a,item['id'],'confusion')
        print('ANALYSIS',a,i+1,len(ds),item['id'],flush=True)
    arrays={k:np.asarray(v) for k,v in data.items()}; arrays['ids']=np.array(ids)
    np.savez_compressed(out/'trajectory_counts.npz',**arrays)
    summaries={}
    for step in [1,2,4]:
        cm=arrays[f's{step}_cm'].sum(0)
        summaries[str(step)]={**G.classification_metrics(cm),'multi_to_few_rate':float(cm[3,2]/cm[3].sum()),
                              'boundary_f1':{str(d):G.summarize_boundaries(arrays[f's{step}_bf{d}'].sum(0)) for d in [1,2,4]},
                              'boundary_iou':{str(d):boundary_iou_score(arrays[f's{step}_biou{d}'].sum(0)) for d in [1,2,4]}}
    G.atomic_json(out/'trajectory_metrics.json',summaries)
    if a=='I':
        metrics,mats,_=C.evaluate(model,c,sampling_seed=3408)
        G.atomic_json(out/'sampling3408.json',metrics); np.savez_compressed(out/'sampling3408_counts.npz',**mats)
    if a in ['N','F']:
        item=ds[0]
        x,_=C.predict(model,c,item['image'],item['thumb'],item['id'],sampling_seed=3407)
        y,_=C.predict(model,c,item['image'],item['thumb'],item['id'],sampling_seed=3408)
        assert np.array_equal(x,y)
        G.atomic_json(out/'seed_invariance.json',dict(exact=True,id=item['id'],sampling_seeds=[3407,3408]))
    if a=='R':
        for name in ['zero_state','zero_selfcond']:
            m,counts,_=C.evaluate(model,c,intervention=name)
            G.atomic_json(out/f'{name}.json',dict(metrics=m,intervention_at_t=.25,scope='fixed weight OOD input intervention; no causal claim from this alone'))
            np.savez_compressed(out/f'{name}_counts.npz',**counts)
    G.atomic_json(out/'analysis_complete.json',dict(meta=meta,finished_unix=time.time(),seconds=time.time()-started,images=25,all_final_counts_match=True))

def report():
    records={}; arrays={}; current={}; at100={}
    for a in ['R','I','N','F']:
        out=C.ANALYSIS/a
        records[a]=json.loads((out/'analysis_complete.json').read_text())
        with np.load(out/'trajectory_counts.npz') as z: arrays[a]={k:z[k].copy() for k in z.files}
        current[a]=json.loads((out/'trajectory_metrics.json').read_text())
        at100[a]=json.loads((C.RUNS[a]/'validation_epoch100.json').read_text())
    for a in arrays: assert np.array_equal(arrays['R']['ids'],arrays[a]['ids'])
    ix=np.random.default_rng(20261031).integers(0,25,size=(10000,25)); boot={}; point={}
    for a,d in arrays.items():
        boot[a]=array_scores(d['s4_cm'][ix].sum(1),d['s4_bf2'][ix].sum(1))
        point[a]={k:float(v) for k,v in array_scores(d['s4_cm'].sum(0),d['s4_bf2'].sum(0)).items()}
    contrasts={}
    for x,y in [('R','I'),('R','N'),('N','F'),('N','R')]:
        differences={k:{'point_pp':100*(point[x][k]-point[y][k]),'image_bootstrap_95_interval_pp':(100*np.quantile(boot[x][k]-boot[y][k],[.025,.975])).tolist()} for k in point[x]}
        gate=(differences['boundary_f1']['point_pp']>=(-.5 if (x,y)==('N','R') else 1.) and differences['miou']['point_pp']>=-.3 and differences['thin_mean_iou']['point_pp']>=-.5)
        contrasts[x+'-'+y]=dict(differences=differences,practical_gate_met=gate if (x,y)!=('R','N') else None)
    benches={}
    for a in ['R','I','N','F']:
        benches[a]={str(k):json.loads((C.ANALYSIS/'benchmark'/f'{a}_stop{k}.json').read_text()) for k in [1,2,4]}
    result=dict(scope='single training seed42, historically developed val25; no new test inference',
                statistical_limitations=['Image-level resampling assumes independent images; parent/sample groups unverified','Descriptive post-development intervals, not confirmatory significance or equivalence','No estimate of training-seed variation; no multiplicity correction'],
                records=records,point_fraction=point,contrasts=contrasts,trajectories=current,epoch100=at100,benchmark=benches)
    G.atomic_json(C.ANALYSIS/'comparison_summary.json',result)
    lines=['GLP-Flow受控消融批次完成（seed42，原val25）','',
           '本批完成I/N/F各100轮；R复用历史E3R。无新增test推理。以下差值单位为百分点。','',
           '方法  best轮次  mIoU%  前景边界F1%  薄层均值%  整图四步秒']
    for a in ['R','I','N','F']:
        p=point[a]; lines.append(f"{a:3} {records[a]['meta']['epoch']:6} {100*p['miou']:.4f} {100*p['boundary_f1']:.4f} {100*p['thin_mean_iou']:.4f} {benches[a]['4']['image_predict_s_median']:.4f}")
    lines+=['','预先指定的主要比较：']
    for name in ['R-I','R-N','N-F','N-R']:
        row=contrasts[name]; lines.append(name+': '+json.dumps(row,ensure_ascii=False))
    lines+=['','R-I衡量监督状态来源；R-N衡量初态噪声；N-F衡量当前Euler混合相对端点回馈。',
            '实际收益门槛不等于统计显著性；N-R通过容忍门槛也不证明统计等效。',
            '所有结果均保留，不按结果追加轮数、改变数据或替换原方法。',
            '过程、边界容差、Boundary IoU、epoch100、原始计数和固定K4提前结束测速见comparison_summary.json及各组目录。',
            '图像级bootstrap仅是固定权重/旧划分内的描述；样品独立性未核验，也未进行多训练seed复现。']
    (C.ANALYSIS/'final_report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    if '--report' in sys.argv: report()
    else: arm(sys.argv[sys.argv.index('--arm')+1])

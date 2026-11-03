"""Read-only numerical audit of saved results; no model inference or training."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/analysis/story_audit_20261025'
OUT.mkdir(parents=True, exist_ok=True)
sources = set()

def read_json(p):
    p = ROOT / p
    sources.add(p)
    return json.loads(p.read_text())

def load(p):
    sources.add(p)
    with np.load(p) as f:
        z = {k: f[k] for k in f.files}
    assert np.array_equal(z['global_matrix'], z['per_image'].sum(0)), p
    assert np.array_equal(z['boundary_counts'], z['per_image_boundary'].sum(0)), p
    return z

def iou(m, gt_only=False):
    tp = np.diagonal(m, axis1=-2, axis2=-1)
    gt, pred = m.sum(-1), m.sum(-2)
    union = gt + pred - tp
    out = np.divide(tp, union, out=np.full(tp.shape, np.nan), where=union > 0)
    if gt_only:
        out = np.where(gt > 0, out, np.nan)
    return out

def bf(b):
    p = np.divide(b[..., 0], b[..., 1], out=np.zeros(b.shape[:-1]), where=b[..., 1] > 0)
    r = np.divide(b[..., 2], b[..., 3], out=np.zeros(b.shape[:-1]), where=b[..., 3] > 0)
    f = np.divide(2*p*r, p+r, out=np.zeros_like(p), where=p+r > 0)
    return np.where((b[..., 1]+b[..., 3]) > 0, f, np.nan)

def metric(z):
    m, b = z['global_matrix'], z['boundary_counts']
    v = iou(m) * 100
    f = bf(b) * 100
    return dict(miou=float(v.mean()), iou=v.tolist(), thin=float(v[1:3].mean()),
                bf=float(f[1:].mean()), bf_class=f.tolist(),
                few_precision=100*float(m[2, 2]/m[:, 2].sum()),
                few_recall=100*float(m[2, 2]/m[2].sum()),
                multi_to_few=int(m[3, 2]), few_to_mono=int(m[2, 1]),
                few_to_mono_rate=100*float(m[2, 1]/m[2].sum()))

data = {'val': {}, 'test': {}}
for p in sorted((ROOT/'outputs/P1/glpflow/seed_42').glob('*/best_val_confusion.npz')):
    data['val'][p.parent.name.split('_')[-1].upper()] = load(p)
for p in sorted((ROOT/'outputs/P1/baselines/seed_42').glob('*/best_val_confusion.npz')):
    data['val'][p.parent.name.removeprefix('p1_baselines_20261022_')] = load(p)
data['val']['repela'] = load(next((ROOT/'outputs/P1/repela/seed_42').glob('*/best_val_confusion.npz')))
for folder in ['glpflow_test_20261022', 'baselines_test_20261022']:
    for p in sorted((ROOT/'outputs/inference'/folder).glob('*_confusion.npz')):
        data['test'][p.stem.removesuffix('_confusion')] = load(p)
data['test']['repela'] = load(ROOT/'outputs/inference/repela_test_20261024/train_form_confusion.npz')

result = {'metrics': {}, 'paired': {}, 'sensitivity': {}, 'error_concentration': {}, 'curves': {}}
rows = []
for split, models in data.items():
    result['metrics'][split] = {name: metric(z) for name,z in models.items()}
    ref = models['E3R']
    ids = ref['ids'].tolist()
    for name,z in models.items():
        assert z['ids'].tolist() == ids, (split, name)
        assert np.array_equal(z['per_image'].sum(-1), ref['per_image'].sum(-1)), (split,name)
        v = metric(z)
        rows.append(dict(split=split, model=name, miou=v['miou'], thin=v['thin'], boundary_f1=v['bf'],
                         **{f'iou_{i}': x for i,x in enumerate(v['iou'])}))
    result['paired'][split] = {}
    result['sensitivity'][split] = {}
    for name in ['segformer_b1','E0','E1','E2','E3','mask2former','ddp','repela']:
        z = models[name]
        a, b = ref['per_image'], z['per_image']
        dm = 100*(np.nanmean(iou(a), -1)-np.nanmean(iou(b), -1))
        dm_gt = 100*(np.nanmean(iou(a,True), -1)-np.nanmean(iou(b,True), -1))
        db = 100*(np.nanmean(bf(ref['per_image_boundary'])[:,1:], -1)-np.nanmean(bf(z['per_image_boundary'])[:,1:], -1))
        def stats(d):
            return dict(wins=int((d>1e-10).sum()), losses=int((d < -1e-10).sum()),
                        median=float(np.median(d)), mean=float(d.mean()),
                        min=float(d.min()), max=float(d.max()))
        result['paired'][split][name] = dict(miou=stats(dm), miou_gt_present=stats(dm_gt), bf=stats(db),
            images=[dict(id=id, miou_delta=float(x), miou_gt_present_delta=float(y), bf_delta=float(f))
                    for id,x,y,f in zip(ids,dm,dm_gt,db)])
        keep_sets = {'all': list(range(len(ids)))}
        if split == 'test':
            keep_sets.update({f'without_{id}':[i for i,v in enumerate(ids) if v!=id] for id in ['m122','m150']})
            keep_sets['without_m122_m150'] = [i for i,v in enumerate(ids) if v not in ['m122','m150']]
        else:
            keep_sets['without_m109']=[i for i,v in enumerate(ids) if v!='m109']
        sens = {}
        for tag,keep in keep_sets.items():
            am,bm=a[keep].sum(0),b[keep].sum(0)
            ab,bb=ref['per_image_boundary'][keep].sum(0),z['per_image_boundary'][keep].sum(0)
            sens[tag]=dict(miou_delta=100*float(np.nanmean(iou(am))-np.nanmean(iou(bm))),
                           bf_delta=100*float(np.nanmean(bf(ab)[1:])-np.nanmean(bf(bb)[1:])))
        loo = [100*float(np.nanmean(iou(a.sum(0)-a[i]))-np.nanmean(iou(b.sum(0)-b[i]))) for i in range(len(ids))]
        loo_bf=[100*float(np.nanmean(bf(ref['boundary_counts']-ref['per_image_boundary'][i])[1:])-np.nanmean(bf(z['boundary_counts']-z['per_image_boundary'][i])[1:])) for i in range(len(ids))]
        sens['leave_one_out_range']=dict(miou=[min(loo),max(loo)],bf=[min(loo_bf),max(loo_bf)])
        result['sensitivity'][split][name]=sens
    result['error_concentration'][split]={}
    for name in ['E0','E1','E3R','segformer_b1','repela']:
        z=models[name]; m=z['per_image']; order=np.argsort(m[:,3,2])[::-1]
        result['error_concentration'][split][name]=dict(
            multi_to_few_top=[dict(id=ids[i],pixels=int(m[i,3,2])) for i in order[:5]],
            few_to_mono_top=[dict(id=ids[i],pixels=int(m[i,2,1])) for i in np.argsort(m[:,2,1])[::-1][:5]],
            gt_few_top=[dict(id=ids[i],pixels=int(m[i,2].sum())) for i in np.argsort(m[:,2].sum(-1))[::-1][:5]])

for p in sorted((ROOT/'outputs/P1').glob('*/seed_42/*/metrics_val.csv')):
    if p.parent.parent.parent.name not in ['glpflow','baselines','repela']: continue
    sources.add(p)
    rs=list(csv.DictReader(p.open()))
    points={int(r['epoch']):float(r['value'])*100 for r in rs if r['tag']=='val/miou_4class'}
    if points:
        result['curves'][p.parent.name]=dict(best_epoch=max(points,key=points.get),points=points,
            last10_gain=points.get(100,0)-points.get(90,0))

result['benchmark_old']=read_json('docs/figures/chapter4_20261024/benchmark.json')
result['benchmark_current']=read_json('outputs/analysis/ch4_benchmark_rerun_20261024/benchmark.json')
result['benchmark_revision']=read_json('outputs/analysis/ch4_benchmark_rerun_20261024/host_load.json')
result['process']=read_json('docs/figures/chapter4_20261024/process_statistics.json')
result['dataset']=read_json('data_manifests/full_audit.json')
result['note']='Descriptive reanalysis of the fixed split and seed42. No retraining, new inference, checkpoint selection, significance/equivalence claim, or deletion of difficult images.'
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
with (OUT/'metrics_recomputed.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'sources.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},indent=2))
print('Saved',OUT)
for split in ['val','test']:
    print('\n',split)
    for name in ['E0','E1','E2','E3','E3R','segformer_b1','mask2former','ddp','repela']:
        v=result['metrics'][split][name]
        print(name,'mIoU/BF/fewIoU/fewPrecision/fewRecall',*[round(v[k],4) for k in ['miou','bf']],round(v['iou'][2],4),round(v['few_precision'],4),round(v['few_recall'],4))
    for name in ['segformer_b1','E0','E1','E3']:
        print('paired E3R -',name, {k:v for k,v in result['paired'][split][name].items() if k!='images'})
        print('sensitivity',result['sensitivity'][split][name])

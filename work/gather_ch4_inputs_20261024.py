import json, shutil, tarfile, hashlib
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/ConditionalDiffusionSeg')
O=R/'work/ch4_paper_inputs_20261024'
O.mkdir(exist_ok=True)
files=[]
def cp(p):
 p=Path(p); rel=p.relative_to(R); q=O/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q); files.append(str(rel))
for rel in ['data_manifests','docs/CHAPTER4_FIGURES_TABLES_AND_TRAINING_PLAN_20261024.md','docs/CHAPTER3_WRITING_BLUEPRINT_20261023.md','docs/TRAINING_PLAN_BASELINES_20261022.md','docs/figures/glpflow_e3r_cvpr_20261023','docs/figures/glpflow_e3r_20261022','docs/thesis/chapter3_20261019/scripts/font_config.py']:
 p=R/rel
 if p.is_file(): cp(p)
 else:
  for f in p.rglob('*'):
   if f.is_file() and f.suffix in ['.json','.md','.txt','.csv','.py','.svg','.pdf','.png','.ttf','.otf']: cp(f)
for rel in ['outputs/launches/p1_baselines_20261022','outputs/launches/p1_glpflow_20261021','outputs/launches/p1_glpflow_rollout_20261022','outputs/launches/p1_ch4_extension_20261024','outputs/analysis/glpflow_diagnosis_20261022','outputs/analysis/glpflow_e3r_mechanism_20261022','outputs/analysis/e3r_process_export_20261024','outputs/inference/glpflow_test_20261022','outputs/inference/baselines_test_20261022','outputs/inference/repela_test_20261024']:
 for f in (R/rel).rglob('*'):
  if f.is_file() and f.suffix in ['.json','.npz','.csv','.md']: cp(f)
for f in (R/'outputs/P1').rglob('metrics_val.csv'):
 if any(s in str(f) for s in ['/glpflow/','/baselines/','/repela/']): cp(f)
for f in (R/'outputs/P1').rglob('best_val_confusion.npz'):
 if any(s in str(f) for s in ['/glpflow/','/baselines/','/repela/']): cp(f)
man=json.loads((R/'data_manifests/source_dataset.json').read_text()); D=Path(man['dataset_root'])
val=Path(man['splits']['val']['snapshot_path']).read_text().split()
test=Path(man['splits']['test']['snapshot_path']).read_text().split()
def miou(c):
 c=c.astype(float); a=np.diagonal(c,axis1=-2,axis2=-1); u=c.sum(-1)+c.sum(-2)-a
 return np.nanmean(np.divide(a,u,out=np.full_like(a,np.nan),where=u>0),axis=-1)
z=np.load(R/'outputs/P1/glpflow/seed_42/p1_glpflow_rollout_20261022_e3r/best_val_confusion.npz')
c=z['per_image']; fr=c.sum(-1)/c.sum((-1,-2))[:,None]; scores=miou(c)
candidate=[(str(i),float(s)) for i,s,f in zip(z['ids'],scores,fr) if f[1]>=.015 and f[2]>=.015]
med=float(np.median([s for _,s in candidate])); selected=[i for i,s in sorted(candidate,key=lambda a:(abs(a[1]-med),a[0]))[:2]]
t=np.load(R/'outputs/inference/glpflow_test_20261022/E3R_confusion.npz'); tf=t['per_image'].sum(-1)/t['per_image'].sum((-1,-2))[:,None]
# Prespecified class-composition rule, independent of relative model performance.
tselected=list(dict.fromkeys([str(t['ids'][np.argmax(tf[:,1])]),str(t['ids'][np.argmax(tf[:,2])])]))
sel=dict(process_rule='val images with monolayer and fewlayer >=1.5%; two closest to candidate median E3R per-image mIoU; lexicographic tie break',candidates=candidate,candidate_median=med,process_selected=selected,test_rule='test image with largest monolayer fraction and image with largest fewlayer fraction, no method-score selection',test_selected=tselected)
(O/'selection.json').write_text(json.dumps(sel,ensure_ascii=False,indent=2))
for split,ids in [('val',val),('test',tselected)]:
 for i in ids:
  for key,ext in [('image_pattern','.jpg'),('mask_pattern','.png')]:
   p=D/man[key].format(id=i); q=O/'dataset'/split/('rgb' if ext=='.jpg' else 'gt')/(i+ext);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for i in tselected:
 for method in ['segformer_b1','mask2former','ddp','E0','E3R']:
  group='glpflow' if method.startswith('E') else 'baselines'
  p=R/f'outputs/inference/{group}_test_20261022/{method}/masks/{i}.png';cp(p)
# Existing publication fonts, without changing the host font configuration.
fontroot=R/'docs/thesis/chapter3_20261019'
for p in fontroot.rglob('*'):
 if p.is_file() and p.suffix in ['.ttf','.otf']:cp(p)
(O/'input_manifest.json').write_text(json.dumps({'files':files,'selection':sel},ensure_ascii=False,indent=2))
archive=R/'work/ch4_paper_inputs_20261024.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as t:t.add(O,arcname='inputs')
print(json.dumps({'archive':str(archive),'size_mib':archive.stat().st_size/2**20,'selection':sel},ensure_ascii=False),flush=True)

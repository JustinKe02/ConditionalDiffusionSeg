"""Read-only code/result/manuscript audit; writes only a new dated audit directory."""
from pathlib import Path
import ast, csv, hashlib, json, re, shutil, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/analysis/workspace_review_20261103';OUT.mkdir(parents=True,exist_ok=True)
AN=ROOT/'outputs/analysis/chapter_controlled_ablation_20261031'
DOC=ROOT/'docs/thesis/chapters35_integration_20261101/masters_thesis_20261101'
S=json.loads((AN/'comparison_summary.json').read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def score(cm,bc):
 tp=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(-1)+cm.sum(-2)-tp
 iou=np.divide(tp,den,out=np.full(tp.shape,np.nan),where=den>0)
 p=np.divide(bc[...,0],bc[...,1],out=np.zeros(bc.shape[:-1]),where=bc[...,1]>0)
 r=np.divide(bc[...,2],bc[...,3],out=np.zeros(bc.shape[:-1]),where=bc[...,3]>0)
 f=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r>0)
 f=np.where(bc[...,1]+bc[...,3]>0,f,np.nan)
 return dict(miou=np.nanmean(iou,-1),boundary_f1=np.nanmean(f[...,1:],-1),thin_mean_iou=np.nanmean(iou[...,1:3],-1))
report={'date':'2026-11-03','scope':'CPU/read-only verification, no new training or inference','has_git':(ROOT/'.git').exists()}
plan=json.loads((ROOT/'outputs/launches/chapter_controlled_ablation_20261031/execution_plan.json').read_text())
report['frozen_sources']={'count':len(plan['frozen_sha256']),'mismatches':[p for p,h in plan['frozen_sha256'].items() if not (ROOT/p).exists() or digest(ROOT/p)!=h]}
dat=json.loads((ROOT/'outputs/launches/chapter_controlled_ablation_20261031/train_val_data_sha256.json').read_text())
report['frozen_data']={'count':len(dat),'mismatches':[p for p,h in dat.items() if not Path(p).exists() or digest(Path(p))!=h]}
arrays={};boot={};rep={}
ix=np.random.default_rng(20261031).integers(0,25,(10000,25))
for a in ['R','I','N','F']:
 with np.load(AN/a/'trajectory_counts.npz') as z:d={k:z[k] for k in z.files}
 arrays[a]=d;assert np.array_equal(arrays['R']['ids'],d['ids'])
 errors=[]
 for step in ['1','2','4']:
  for width in ['1','2','4']:
   got=score(d[f's{step}_cm'].sum(0),d[f's{step}_bf{width}'].sum(0))
   m=S['trajectories'][a][step]
   errors.extend([abs(got['miou']-m['miou_4class']),abs(got['boundary_f1']-m['boundary_f1'][width]['boundary_f1_foreground'])])
   bi=d[f's{step}_biou{width}'].sum(0);v=np.mean(bi[1:,0]/bi[1:,1]);errors.append(abs(v-m['boundary_iou'][width]['foreground']))
 boot[a]=score(d['s4_cm'][ix].sum(1),d['s4_bf2'][ix].sum(1))
 run=ROOT/'outputs/P1/glpflow/seed_42'/('p1_glpflow_rollout_20261022_e3r' if a=='R' else f'chapter_controlled_20261031_{a.lower()}')
 config=json.loads((run/'config.json').read_text());status=json.loads((run/'status.json').read_text())
 rep[a]={'max_metric_absolute_error':float(max(errors)),'status':status,'best_epoch':S['records'][a]['meta']['epoch'],'checkpoint_hash_matches':digest(Path(S['records'][a]['meta']['checkpoint']))==S['records'][a]['meta']['checkpoint_sha256']}
report['arms']=rep
cis=[]
for con,v in S['contrasts'].items():
 x,y=con.split('-')
 for k,m in v['differences'].items():
  est=100*np.quantile(boot[x][k]-boot[y][k],[.025,.975]);cis.append(float(np.max(np.abs(est-m['image_bootstrap_95_interval_pp']))))
report['bootstrap_max_absolute_error_pp']=max(cis)
# Sensitivity is descriptive; leave-one-out values are not a new test set.
loo={};deltas=[]
for con in ['R-I','R-N','N-F']:
 x,y=con.split('-');vals={}
 for a in [x,y]:
  d=arrays[a];vals[a]=score(d['s4_cm'].sum(0)[None]-d['s4_cm'],d['s4_bf2'].sum(0)[None]-d['s4_bf2'])
 loo[con]={k:{'min_pp':float(np.min((vals[x][k]-vals[y][k])*100)),'max_pp':float(np.max((vals[x][k]-vals[y][k])*100))}for k in vals[x]}
 px=score(arrays[x]['s4_cm'],arrays[x]['s4_bf2']);py=score(arrays[y]['s4_cm'],arrays[y]['s4_bf2'])
 for idx,ident in enumerate(arrays[x]['ids']):deltas.append({'contrast':con,'id':str(ident),**{k+'_delta_pp':float((px[k][idx]-py[k][idx])*100) for k in px}})
report['leave_one_image_out']=loo
with (OUT/'per_image_descriptive_differences.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=deltas[0]);w.writeheader();w.writerows(deltas)
T=json.loads((DOC/'tables/tables.json').read_text());checks=[]
for row in T['t4_best']['rows']:
 a=row[0];expected=[S['point_fraction'][a][k]*100 for k in ['miou','boundary_f1','thin_mean_iou']]
 checks.append(row[1]==S['records'][a]['meta']['epoch'] and all(f'{v:.4f}'==got for v,got in zip(expected,row[2:])))
for row in T['t4_epoch100']['rows']:
 expected=[S['epoch100'][row[0]][k]*100 for k in ['miou_4class','boundary_f1_foreground','thin_mean_iou']]
 checks.append(all(f'{v:.4f}'==got for v,got in zip(expected,row[1:])))
figs=[];missing=[];chapter4=(DOC/'chapters/04_flow.md').read_text()
for p in (DOC/'chapters').glob('*.md'):
 for ln in p.read_text().splitlines():
  if ln.startswith('@figure|'):
   fields=ln.split('|',4);figs.append(fields[1]);
   if not (DOC/fields[2]).exists():missing.append(fields[2])
  if ln.startswith('@table|') and ln.split('|')[1] not in T:missing.append(ln)
report['latest_thesis']={'path':str(DOC.relative_to(ROOT)),'controlled_table_values_match':all(checks),'figure_count':len(figs),'missing_references':missing,'chapter4_characters':len(chapter4),'chapter4_figures':chapter4.count('@figure|'),'chapter4_tables':chapter4.count('@table|'),'chapter4_equations':chapter4.count('@eq|')}
# Parse active files without importing launchers or generating bytecode.
pyfiles=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'work').glob('*.py'))
for folder in ['chapter_controlled_20261031','baselines_20261022','repela_20261024']:pyfiles+=list((ROOT/'work'/folder).rglob('*.py'))
syntax=[]
for p in pyfiles:
 try:ast.parse(p.read_text(),filename=str(p))
 except Exception as e:syntax.append({'path':str(p.relative_to(ROOT)),'error':str(e)})
report['syntax']={'files':len(pyfiles),'errors':syntax}
report['paper_variants']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted((ROOT/'docs/thesis').rglob('chapters/04_flow.md'))]
report['document_build_findings']=[
 {'file':str((DOC/'scripts/revise_chapter4_assets.py').relative_to(ROOT)), 'line':12,
  'problem':'ROOT=O.parents[2] resolves to project/docs in this nested copy; required source JSON is absent.'},
 {'file':str((DOC/'scripts/chapter5_plots.py').relative_to(ROOT)), 'line':8,
  'problem':'Output directory is hard-coded to the older chapter5_revision_20261101 copy.'},
 {'file':'work/chapter_controlled_20261031/benchmark.py', 'line':22,
  'problem':'Agreement checks cover m108 only. The latest integrated chapter explicitly states this scope; not an outstanding manuscript error.'}]
report['leave_one_out_note']='R-I mIoU difference becomes +0.54447156 pp when m109 is omitted. Sensitivity only; the primary evaluation retains all 25 images.'
report['checks_passed']=not(report['frozen_sources']['mismatches'] or report['frozen_data']['mismatches'] or syntax or missing) and all(checks) and max(cis)<1e-10 and all(v['checkpoint_hash_matches'] and v['max_metric_absolute_error']<1e-12 for v in rep.values())
log=Path('/tmp/conditional_diffusion_code_audit_20261103_tests.log')
if log.exists():
 shutil.copyfile(log,OUT/'unit_tests.log');report['unit_tests']={'summary':log.read_text().split('----------------------------------------------------------------------')[-1].strip(),'scope':'Existing tests cover primarily historical src/ paths; does not replace GLP CUDA verification.'}
(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
assert report['checks_passed']

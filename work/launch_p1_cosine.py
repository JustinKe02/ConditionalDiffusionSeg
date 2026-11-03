"""Recheck recorded P0 weights, benchmark full val, then run three frozen P1 pilots."""
import os,sys,json,hashlib,subprocess,fcntl,time,traceback,signal
from pathlib import Path
from datetime import datetime,timezone
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.runtime import write_json,source_hashes,directory_size
BATCH_ID='p1_cosine_20261017';BATCH=ROOT/'outputs/launches'/BATCH_ID
OLD=ROOT/'outputs/launches/diffusion_resume_20261017/p0_gate.json'
COS=ROOT/'outputs/P0/diffusion/seed_42/p0_ablations_20261017_cosine_lr'
STAGE='preflight'

def status(value,**extra):write_json(BATCH/'status.json',dict(status=value,stage=STAGE,pid=os.getpid(),updated_at_utc=datetime.now(timezone.utc).isoformat(),**extra))
def command(*args):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def recheck_p0(frozen):
 import torch,numpy as np
 from conditional_diffusion.runtime import seed_all,StorageGuard
 from conditional_diffusion.data import FixedCropDataset,CLASSES
 from conditional_diffusion.model import make_model
 from conditional_diffusion.diffusion import GaussianDiffusion
 from conditional_diffusion.inference import evaluate
 old=json.loads(OLD.read_text());cos=json.loads((COS/'p0_gate.json').read_text())
 assert cos['passed'] and cos['source_sha256']==frozen
 candidates=[next(r for r in old['results'] if r['method']==m) for m in ['direct','iterative']]
 candidates.append(dict(method='diffusion',checkpoint=str(COS/'checkpoints/best.pt'),checkpoint_sha256=cos['checkpoint_sha256']))
 results=[]
 for item in candidates:
  checkpoint=Path(item['checkpoint']);assert sha(checkpoint)==item['checkpoint_sha256']
  run=checkpoint.parent.parent;assert json.loads((run/'status.json').read_text())['status']=='completed'
  state=torch.load(checkpoint,map_location='cpu',weights_only=False);c=state['config']
  assert c['phase']=='P0' and c['debug'] and c['method']==item['method'] and c['fixed_crops']
  assert c.get('timestep_sampling','uniform')=='uniform'
  torch.set_num_threads(c['cpu_threads']);seed_all(c['seed']);StorageGuard(ROOT,run,c['storage']).check(16*1024**2)
  ds=FixedCropDataset(c['manifest'],c['fixed_crops']);model=make_model(c).cuda().eval();model.load_state_dict(state['model'])
  diffusion=GaussianDiffusion(c['timesteps'],'cuda')
  for seed in ([3407,3408] if item['method']=='diffusion' else [3407]):
   metrics,_,_=evaluate(model,diffusion,ds,dict(c,sampling_seed=seed,preview_count=0))
   minimum=min(metrics['iou_'+name] for name in CLASSES)
   r=dict(method=item['method'],sampling_seed=seed,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
          trained_source_sha256=json.loads((run/'run_metadata.json').read_text())['source_sha256'],
          metrics=metrics,minimum_class_iou=minimum,passed=metrics['miou_4class']>=.90 and minimum>=.75)
   results.append(r);print('P0_RECHECK',item['method'],seed,metrics['miou_4class'],minimum,flush=True)
  del model,state;torch.cuda.empty_cache()
 gate=dict(passed=all(r['passed'] for r in results),thresholds=dict(miou_4class=.90,minimum_class_iou=.75),
  results=results,source_sha256=frozen,fixed_crops_sha256=sha(ROOT/'data_manifests/p0_fixed_crops.json'),
  provenance_note='Fresh current-code evaluation of original passed direct/iterative checkpoints and uniform-t cosine diffusion checkpoint; original failed gate retained unchanged. No P0/P1 weight transfer.',
  scope='Fixed TRAIN crop diagnostic; independent diffusion sampling; not held-out validation performance')
 write_json(BATCH/'p0_gate.json',gate)
 if not gate['passed']:raise RuntimeError('Fresh P0 gate failed; do not launch P1')

def benchmark(frozen):
 cached=BATCH/'benchmark_summary.json'
 if cached.exists():
  report=json.loads(cached.read_text())
  if report['passed'] and report['source_sha256']==frozen and report['checkpoint_sha256']==sha(COS/'checkpoints/best.pt'):
   return report
 before=set((COS/'evaluations').glob('val_*'))
 with (BATCH/'full_val_benchmark.log').open('a') as log:
  subprocess.run([sys.executable,'-u','scripts/evaluate.py','--checkpoint',str(COS/'checkpoints/best.pt'),
   '--split','val','--full-val','--window-size','512','--stride','384','--steps','20','--samples','1','--sampling-seed','3407'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 added=set((COS/'evaluations').glob('val_*'))-before
 if len(added)!=1:raise RuntimeError('Cannot identify unique full-val evaluation')
 evaluation=added.pop();metrics=json.loads((evaluation/'metrics.json').read_text());ec=json.loads((evaluation/'evaluation_config.json').read_text());c=ec['config']
 passed=(metrics['image_count']==25 and metrics['nfe_total']==17500 and c['crop_size']==512 and c['stride']==384 and c['sampling_steps']==20 and c['samples_per_image']==1 and c['sampling_seed']==3407 and ec['split']=='val' and ec['source_sha256']==frozen)
 report=dict(passed=passed,evaluation_directory=str(evaluation),checkpoint_sha256=sha(COS/'checkpoints/best.pt'),source_sha256=frozen,
  **{k:metrics[k] for k in ['image_count','nfe_total','seconds','peak_allocated_mib','peak_reserved_mib']},
  seconds_per_image=metrics['seconds']/metrics['image_count'],four_pilot_diffusion_validations_seconds=4*metrics['seconds'],
  scope='Complete val timing only with P0 cosine weights; scores excluded from research summaries')
 write_json(cached,report);print('FULL_VAL_BENCHMARK',json.dumps(report),flush=True)
 if not passed:raise RuntimeError('Full validation benchmark protocol/count mismatch')
 return report

def main():
 global STAGE
 BATCH.mkdir(parents=True,exist_ok=True)
 lock=open(ROOT/'outputs/.experiments.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  status('running')
  frozen=source_hashes(ROOT)
  previous=json.loads((ROOT/'outputs/launches/p0_ablations_20261017/plan.json').read_text())
  if frozen!=previous['source_sha256']:raise RuntimeError('Code/config changed since validated ablations; review before P1')
  pilot=json.loads((ROOT/'configs/pilot.json').read_text())
  assert pilot['phase']=='P1' and not pilot['debug'] and pilot['epochs']==40 and pilot['crop_size']==512 and pilot['batch_size']==2 and pilot['min_lr']==1e-6 and pilot['warmup_epochs']==5
  plan=dict(batch_id=BATCH_ID,source_sha256=frozen,pilot_config=pilot,methods=['direct','iterative','diffusion'],
   initialization='From scratch, seed42 for each method; no P0 weights loaded into P1',
   p0_evidence=[str(OLD),str(COS/'p0_gate.json')],scope='Fresh P0 recheck, full 25-image val timing, then exactly three 40-epoch P1 runs. No test, formal matrix, or high-noise P1.')
  if (BATCH/'plan.json').exists() and json.loads((BATCH/'plan.json').read_text())!=plan:raise RuntimeError('Frozen P1 plan differs')
  write_json(BATCH/'plan.json',plan)
  command(ROOT/'scripts/check_environment.py','--output',BATCH/'environment.json')
  with (BATCH/'storage_preflight.json').open('w') as log:subprocess.run([sys.executable,'scripts/check_storage.py','--runs','3','--parameters','5132676'],stdout=log,check=True)
  for file in ['pipeline_verification.json','high_noise_resume.json']:
   assert json.loads((ROOT/'outputs/launches/p0_ablations_20261017'/file).read_text())['status']=='passed'
  write_json(BATCH/'prior_validation.json',dict(source_sha256=frozen,unit_tests=13,unit_test_log='outputs/launches/p0_ablations_20261017/console.log',pipeline='passed',stratified_resume='passed',note='Same source/config hashes; reuse already completed checks.'))
  STAGE='p0_recheck';status('running');recheck_p0(frozen)
  STAGE='full_val_benchmark';status('running');benchmark(frozen)
  results=[]
  for method in ['direct','iterative','diffusion']:
   if source_hashes(ROOT)!=frozen:raise RuntimeError('Code changed during P1 batch')
   STAGE='p1_'+method;status('running')
   run=ROOT/'outputs/P1'/method/'seed_42'/BATCH_ID;expected=dict(pilot,method=method,seed=42)
   if run.exists():
    if json.loads((run/'config.json').read_text())!=expected:raise RuntimeError('Pilot run config mismatch')
    if json.loads((run/'run_metadata.json').read_text())['source_sha256']!=frozen:raise RuntimeError('Pilot run source mismatch')
    if json.loads((run/'status.json').read_text())['status']!='completed':command(ROOT/'scripts/train.py','--resume',run/'checkpoints/last.pt')
   else:command(ROOT/'scripts/train.py','--config',ROOT/'configs/pilot.json','--method',method,'--seed','42','--run-id',BATCH_ID)
   rs=json.loads((run/'status.json').read_text());assert rs['status']=='completed' and rs['last_completed_epoch']==40 and rs['global_step']==2400
   results.append(dict(method=method,run=str(run),status=rs,best_val=json.loads((run/'best_val_metrics.json').read_text()),run_size_mib=directory_size(run)/1024**2))
   write_json(BATCH/'pilot_partial_results.json',dict(results=results,scope='P1 validation-only results'))
  write_json(BATCH/'pilot_summary.json',dict(results=results,source_sha256=frozen,scope='P1 seed42 validation pilot; no test evaluation'))
  STAGE='finished';status('completed')
 except BaseException as exc:
  status('failed',error=str(exc));traceback.print_exc();raise

if __name__=='__main__':main()

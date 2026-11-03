"""One registered dual-head P1 on the high-noise recycling recipe, then seed3408."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,json,csv,hashlib,subprocess,fcntl,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.runtime import source_hashes,write_json
BATCH_ID='p1_dual_head_20261018';BATCH=ROOT/'outputs/launches'/BATCH_ID
BASE=ROOT/'outputs/P1/diffusion/seed_42/p1_recycling_20261017'
RUN=ROOT/'outputs/P1/diffusion/seed_42'/BATCH_ID
STAGE='preflight'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(*args):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
def status(value,**extra):write_json(BATCH/'status.json',dict(status=value,stage=STAGE,pid=os.getpid(),updated_at_utc=datetime.now(timezone.utc).isoformat(),**extra))
def verify(plan):
    assert source_hashes(ROOT)==plan['source_sha256'],'Frozen source/config changed'
    for category in ['protected_weights','workflow_sha256','historical_reports_sha256']:
        assert all(sha(ROOT/p)==h for p,h in plan[category].items()),category+' changed'

def evaluate3408(plan):
    verify(plan);checkpoint=RUN/'checkpoints/best.pt';digest=sha(checkpoint);cache=BATCH/'candidate_seed3408.json'
    protocol=dict(crop_size=512,stride=384,sampling_steps=20,samples_per_image=1,sampling_seed=3408)
    if cache.exists():
        r=json.loads(cache.read_text());assert r['checkpoint_sha256']==digest and r['protocol']==protocol and r['evaluation_source_sha256']==plan['source_sha256'];return r
    before=set((RUN/'evaluations').glob('val_*'))
    with (BATCH/'candidate_seed3408.log').open('a') as log:
        subprocess.run([sys.executable,'-u','scripts/evaluate.py','--checkpoint',str(checkpoint),'--split','val','--full-val',
                        '--window-size','512','--stride','384','--steps','20','--samples','1','--sampling-seed','3408'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    added=set((RUN/'evaluations').glob('val_*'))-before;assert len(added)==1;out=added.pop()
    m=json.loads((out/'metrics.json').read_text());ec=json.loads((out/'evaluation_config.json').read_text())
    assert m['image_count']==25 and m['nfe_total']==17500
    assert ec['split']=='val' and all(ec['config'][k]==v for k,v in protocol.items()) and ec['source_sha256']==plan['source_sha256']
    assert sha(checkpoint)==digest
    r=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,checkpoint_epoch=ec['checkpoint_epoch'],protocol=protocol,
           evaluation_directory=str(out),metrics=m,evaluation_source_sha256=plan['source_sha256'])
    write_json(cache,r);verify(plan);return r

def main():
    global STAGE
    lock=open(ROOT/'outputs/.experiments.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        frozen=source_hashes(ROOT);before=json.loads((BATCH/'source_before.json').read_text())
        oldplan=json.loads((ROOT/'outputs/launches/p1_uniform_recycling_20261017/plan.json').read_text());assert before==oldplan['source_sha256']
        changed={k for k in set(before)|set(frozen) if before.get(k)!=frozen.get(k)}
        assert changed=={'src/conditional_diffusion/'+p for p in ['model.py','diffusion.py','recycling.py','objectives.py','trainer.py']}|{'configs/pilot_dual_head.json'}
        c=json.loads((ROOT/'configs/pilot_dual_head.json').read_text());bc=json.loads((BASE/'config.json').read_text())
        diff={k:[bc.get(k),c.get(k)] for k in set(bc)|set(c) if bc.get(k)!=c.get(k)};assert set(diff)=={'dual_head','experiment_variant'}
        assert c['seed']==42 and c['epochs']==40 and c['batch_size']==2
        weights=dict(oldplan['protected_checkpoint_sha256'])
        ur=ROOT/'outputs/P1/diffusion/seed_42/p1_uniform_recycling_20261017/checkpoints/best.pt'
        ur_report=json.loads((ROOT/'outputs/launches/p1_uniform_recycling_20261017/candidate_seed3408.json').read_text());assert sha(ur)==ur_report['checkpoint_sha256']
        weights[str(ur)]=sha(ur)
        assert all(sha(p)==h for p,h in weights.items())
        workflows=['work/launch_p1_dual_head.py','work/verify_p1_dual_head_resume.py','work/verify_dual_head_legacy.py']
        workflows += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_*.py'))]
        workflows += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'work/backups/p1_dual_head').glob('*.py'))]
        history=[str((BASE/'best_val_metrics.json').relative_to(ROOT)),'outputs/launches/p1_recycling_20261017/candidate_seed3408.json']
        plan=dict(batch_id=BATCH_ID,source_sha256=frozen,source_change_set=sorted(changed),config=c,config_diff=diff,
                  protected_weights={str(Path(p).relative_to(ROOT)):h for p,h in weights.items()},
                  workflow_sha256={p:sha(ROOT/p) for p in workflows},historical_reports_sha256={p:sha(ROOT/p) for p in history},
                  baseline_trained_source_sha256=json.loads((BASE/'run_metadata.json').read_text())['source_sha256'],
                  initialization='From scratch seed42; denoising head is an independent deepcopy of original semantic output head, consuming no extra initialization RNG. Shared initial weights and subsequent RNG sequence match single-head recipe; no old weights loaded for candidate.',
                  head_contract=dict(semantic='0.5CE+0.5foregroundDice; final raw logits used for probability fusion',denoising='1.0MSE; clamped x0 drives recycling and all DDIM state updates',
                                     interpretation='Complete two-head scheme changes gradient routing and final semantic readout; do not attribute any gain solely to gradient decoupling.'),
                  baseline_parameters=5132676,candidate_parameters=5132872,
                  scope='Exactly one high-noise50%+recycling50% dual-head P1,40epochs/2400updates/seed42, then one full-val3408 evaluation. No5step sweep, other factors/seeds, formal run or test.',
                  selection='Full25val,512/384,20DDIM,one sample,seed3407 every10epochs; mIoU4class best.3408 never reselects weights.',
                  reporting='Continuous two-seed deltas versus high-noise recycling single-head; no significance claim or automatic follow-up.')
        pp=BATCH/'plan.json'
        if pp.exists():assert json.loads(pp.read_text())==plan,'Frozen plan changed'
        write_json(pp,plan);verify(plan)
        if (BATCH/'status.json').exists() and json.loads((BATCH/'status.json').read_text())['status']=='completed':return
        status('running')
        if not (BATCH/'preflight_passed.json').exists():
            command(ROOT/'scripts/check_environment.py','--output',BATCH/'environment.json')
            with (BATCH/'storage_preflight.json').open('w') as log:subprocess.run([sys.executable,'scripts/check_storage.py','--runs','1','--parameters','5132872'],cwd=ROOT,stdout=log,check=True)
            with (BATCH/'unit_tests.log').open('w') as log:subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            command(ROOT/'work/verify_dual_head_legacy.py')
            if not (BATCH/'resume_verification.json').exists():command(ROOT/'work/verify_p1_dual_head_resume.py','--output',BATCH/'resume_verification.json')
            assert json.loads((BATCH/'resume_verification.json').read_text())['status']=='passed'
            verify(plan);write_json(BATCH/'preflight_passed.json',dict(status='passed',source_sha256=frozen))
        STAGE='training_dual_head';status('running');verify(plan)
        if RUN.exists():
            assert json.loads((RUN/'config.json').read_text())==c
            assert json.loads((RUN/'run_metadata.json').read_text())['source_sha256']==frozen
            if json.loads((RUN/'status.json').read_text())['status']!='completed':command(ROOT/'scripts/train.py','--resume',RUN/'checkpoints/last.pt')
        else:command(ROOT/'scripts/train.py','--config',ROOT/'configs/pilot_dual_head.json','--run-id',BATCH_ID)
        rs=json.loads((RUN/'status.json').read_text());assert rs['status']=='completed' and rs['global_step']==2400 and rs['last_completed_epoch']==40
        rows=list(csv.DictReader((RUN/'metrics_epochs.csv').open()))
        for tag in ['diffusion_train/high_noise_fraction','diffusion_train/recycle_fraction']:
            vals=[float(r['value']) for r in rows if r['tag']==tag];assert len(vals)==40 and all(v==.5 for v in vals)
        verify(plan);STAGE='candidate_seed3408';status('running');new3408=evaluate3408(plan)
        old3407=json.loads((BASE/'best_val_metrics.json').read_text());new3407=json.loads((RUN/'best_val_metrics.json').read_text())
        old3408=json.loads((ROOT/history[1]).read_text());keys=['miou_4class','miou_foreground','iou_monolayer','iou_fewlayer','boundary_f1_foreground','seconds']
        pairs={'3407':(old3407['metrics'],new3407['metrics']),'3408':(old3408['metrics'],new3408['metrics'])}
        report=dict(baseline_seed3407=old3407,candidate_seed3407=new3407,baseline_seed3408=old3408,candidate_seed3408=new3408,
                    deltas={s:{k:b[k]-a[k] for k in keys} for s,(a,b) in pairs.items()},scope=plan['scope'],head_contract=plan['head_contract'],
                    candidate_parameters=plan['candidate_parameters'],baseline_parameters=plan['baseline_parameters'],
                    optimizer_steps=2400,actual_high_noise_samples=2400,actual_recycled_samples=2400,source_sha256=frozen)
        verify(plan);write_json(BATCH/'comparison_summary.json',report);STAGE='finished';status('completed')
    except BaseException as exc:
        status('failed',error=str(exc));traceback.print_exc();raise

if __name__=='__main__':main()

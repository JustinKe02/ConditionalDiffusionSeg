"""Exactly one high-noise P1 trial, followed by baseline/candidate seed3408 val."""
import os,sys,json,csv,hashlib,subprocess,fcntl,traceback
from pathlib import Path
from datetime import datetime,timezone
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.runtime import write_json,source_hashes
BATCH_ID='p1_high_noise_20261017';BATCH=ROOT/'outputs/launches'/BATCH_ID
BASE=ROOT/'outputs/P1/diffusion/seed_42/p1_cosine_20261017'
RUN=ROOT/'outputs/P1/diffusion/seed_42'/BATCH_ID
STAGE='preflight'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(*args):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
def status(value,**extra):write_json(BATCH/'status.json',dict(status=value,stage=STAGE,pid=os.getpid(),updated_at_utc=datetime.now(timezone.utc).isoformat(),**extra))
def evaluate_seed3408(run,label,frozen):
    checkpoint=run/'checkpoints/best.pt';cache=BATCH/(label+'_seed3408.json')
    protocol=dict(crop_size=512,stride=384,sampling_steps=20,samples_per_image=1,sampling_seed=3408)
    if cache.exists():
        saved=json.loads(cache.read_text())
        if saved['checkpoint_sha256']==sha(checkpoint) and saved['protocol']==protocol and saved['evaluation_source_sha256']==frozen:
            return saved
        raise RuntimeError('Cached seed evaluation differs from frozen protocol/source')
    before=set((run/'evaluations').glob('val_*'))
    with (BATCH/(label+'_seed3408.log')).open('a') as log:
        subprocess.run([sys.executable,'-u','scripts/evaluate.py','--checkpoint',str(checkpoint),'--split','val',
            '--full-val','--window-size','512','--stride','384','--steps','20','--samples','1','--sampling-seed','3408'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    added=set((run/'evaluations').glob('val_*'))-before
    if len(added)!=1:raise RuntimeError('Cannot identify unique evaluation output')
    directory=added.pop();metrics=json.loads((directory/'metrics.json').read_text());ec=json.loads((directory/'evaluation_config.json').read_text())
    assert metrics['image_count']==25 and metrics['nfe_total']==17500
    assert ec['split']=='val' and all(ec['config'][key]==value for key,value in protocol.items())
    assert ec['source_sha256']==frozen
    report=dict(label=label,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),checkpoint_epoch=ec['checkpoint_epoch'],protocol=protocol,
                evaluation_directory=str(directory),metrics=metrics,evaluation_source_sha256=frozen,
                trained_source_sha256=json.loads((run/'run_metadata.json').read_text())['source_sha256'])
    write_json(cache,report);print('SEED3408_RESULT',label,json.dumps(report),flush=True)
    return report

def main():
    global STAGE
    BATCH.mkdir(parents=True,exist_ok=True)
    lock=open(ROOT/'outputs/.experiments.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        status('running')
        frozen=source_hashes(ROOT);before=json.loads((BATCH/'source_before.json').read_text())
        changed={key for key in set(before)|set(frozen) if before.get(key)!=frozen.get(key)}
        if changed!={'src/conditional_diffusion/trainer.py','configs/pilot_high_noise.json'}:raise RuntimeError('Unexpected source changes beyond explicit config validation and new config')
        assert json.loads((BASE/'status.json').read_text())['status']=='completed'
        baseline=json.loads((BASE/'config.json').read_text());config=json.loads((ROOT/'configs/pilot_high_noise.json').read_text())
        diff={key:[baseline.get(key),config.get(key)] for key in set(baseline)|set(config) if baseline.get(key)!=config.get(key)}
        assert set(diff)=={'timestep_sampling','high_noise_start','experiment_variant'}
        assert config['epochs']==40 and config['batch_size']==2 and config['seed']==42
        baseline_hash=sha(BASE/'checkpoints/best.pt')
        plan=dict(batch_id=BATCH_ID,source_sha256=frozen,source_change_set=sorted(changed),config=config,config_diff=diff,
            baseline_run=str(BASE),baseline_checkpoint_sha256=baseline_hash,baseline_trained_source_sha256=json.loads((BASE/'run_metadata.json').read_text())['source_sha256'],
            test_source_sha256=sha(ROOT/'tests/test_p1_high_noise_config.py'),
            initialization='From scratch with seed42; same initialization and RGB crop/order, changed timestep policy also changes subsequent noise RNG sequence.',
            evaluation=dict(split='val',images=25,window=512,stride=384,steps=20,samples=1,checkpoint_selection_seed=3407,extra_seed=3408),
            screening_rule='primary seed3407 mIoU delta >=0.05, and mIoU plus monolayer IoU improve at both seeds; screening only, not significance or final thesis evidence.',
            scope='Exactly one 40-epoch high-noise diffusion P1 and two seed3408 full-val evaluations. No other training, test data or formal matrix.')
        pp=BATCH/'plan.json'
        if pp.exists() and json.loads(pp.read_text())!=plan:raise RuntimeError('Frozen plan or baseline/source changed')
        write_json(pp,plan)
        command(ROOT/'scripts/check_environment.py','--output',BATCH/'environment.json')
        with (BATCH/'storage_preflight.json').open('w') as log:subprocess.run([sys.executable,'scripts/check_storage.py','--runs','1','--parameters','5132676'],stdout=log,check=True)
        command('-m','unittest','discover','-s','tests','-v')
        if not (BATCH/'resume_verification.json').exists():command(ROOT/'work/verify_p1_high_noise_resume.py','--output',BATCH/'resume_verification.json')
        assert json.loads((BATCH/'resume_verification.json').read_text())['status']=='passed'
        STAGE='training_high_noise';status('running')
        if RUN.exists():
            assert json.loads((RUN/'config.json').read_text())==config
            assert json.loads((RUN/'run_metadata.json').read_text())['source_sha256']==frozen
            if json.loads((RUN/'status.json').read_text())['status']!='completed':command(ROOT/'scripts/train.py','--resume',RUN/'checkpoints/last.pt')
        else:command(ROOT/'scripts/train.py','--config',ROOT/'configs/pilot_high_noise.json','--run-id',BATCH_ID)
        rs=json.loads((RUN/'status.json').read_text());assert rs['status']=='completed' and rs['last_completed_epoch']==40 and rs['global_step']==2400
        rows=list(csv.DictReader((RUN/'metrics_epochs.csv').open()));fractions=[float(r['value']) for r in rows if r['tag']=='diffusion_train/high_noise_fraction']
        assert len(fractions)==40 and all(x==.5 for x in fractions)
        if source_hashes(ROOT)!=frozen or sha(BASE/'checkpoints/best.pt')!=baseline_hash:raise RuntimeError('Source/baseline changed during run')
        STAGE='baseline_seed3408';status('running');base3408=evaluate_seed3408(BASE,'baseline',frozen)
        STAGE='candidate_seed3408';status('running');new3408=evaluate_seed3408(RUN,'candidate',frozen)
        old3407=json.loads((BASE/'best_val_metrics.json').read_text());new3407=json.loads((RUN/'best_val_metrics.json').read_text())
        pairs={3407:(old3407['metrics'],new3407['metrics']),3408:(base3408['metrics'],new3408['metrics'])}
        deltas={str(seed):{key:new[key]-old[key] for key in ['miou_4class','miou_foreground','iou_monolayer','iou_fewlayer','boundary_f1_foreground']} for seed,(old,new) in pairs.items()}
        screen=deltas['3407']['miou_4class']>=.05 and all(deltas[str(seed)]['miou_4class']>0 and deltas[str(seed)]['iou_monolayer']>0 for seed in [3407,3408])
        report=dict(baseline_seed3407=old3407,candidate_seed3407=new3407,baseline_seed3408=base3408,candidate_seed3408=new3408,
            deltas=deltas,screen_passed=screen,screening_rule=plan['screening_rule'],actual_high_noise_samples=int(sum(fractions)*120),optimizer_steps=2400,
            source_sha256=frozen,scope=plan['scope'],interpretation='One training seed, two sampling seeds; selection remains fixed at3407. Original baseline and checkpoint preserved.')
        write_json(BATCH/'comparison_summary.json',report)
        STAGE='finished';status('completed',screen_passed=screen)
    except BaseException as exc:
        status('failed',error=str(exc));traceback.print_exc();raise

if __name__=='__main__':main()

"""Run exactly two separately logged P0 diffusion ablations; never advance to P1."""
import os,sys,json,subprocess,fcntl,traceback,hashlib
from pathlib import Path
from datetime import datetime,timezone
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.runtime import write_json,source_hashes,StorageGuard,seed_all
BATCH_ID='p0_ablations_20261017'
BATCH=ROOT/'outputs/launches'/BATCH_ID
PLAN=[('high_noise','configs/p0_high_noise.json'),('cosine_lr','configs/p0_cosine_lr.json')]

def status(value,stage,**extra):
    write_json(BATCH/'status.json',dict(status=value,stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**extra))

def command(*args):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)

def assess(run):
    import torch,numpy as np
    from conditional_diffusion.data import FixedCropDataset,CLASSES
    from conditional_diffusion.model import make_model
    from conditional_diffusion.diffusion import GaussianDiffusion
    from conditional_diffusion.inference import evaluate
    assert json.loads((run/'status.json').read_text())['status']=='completed'
    assert json.loads((run/'run_metadata.json').read_text())['source_sha256']==source_hashes(ROOT)
    state=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    c=state['config'];torch.set_num_threads(c['cpu_threads']);seed_all(c['seed'])
    guard=StorageGuard(ROOT,run,c['storage']);guard.check(16*1024**2)
    ds=FixedCropDataset(c['manifest'],c['fixed_crops'])
    model=make_model(c).cuda().eval();model.load_state_dict(state['model']);diffusion=GaussianDiffusion(c['timesteps'],'cuda')
    results=[]
    for seed in [3407,3408]:
        metrics,matrices,previews=evaluate(model,diffusion,ds,dict(c,sampling_seed=seed))
        minimum=min(metrics['iou_'+name] for name in CLASSES)
        item=dict(sampling_seed=seed,metrics=metrics,minimum_class_iou=minimum,
                  passed=metrics['miou_4class']>=.90 and minimum>=.75)
        results.append(item)
        dest=run/f'p0_sampling_seed_{seed}';dest.mkdir(exist_ok=True)
        write_json(dest/'metrics.json',item);np.savez_compressed(dest/'confusion.npz',**matrices)
        from PIL import Image
        for ident,panels in previews.items():Image.fromarray(np.concatenate(list(panels.values()),axis=1)).save(dest/(ident+'.png'))
        print('GATE',run.name,json.dumps(item),flush=True)
    report=dict(passed=all(x['passed'] for x in results),best_epoch=state['epoch'],results=results,
                thresholds=dict(miou_4class=.90,minimum_class_iou=.75),source_sha256=source_hashes(ROOT),
                checkpoint_sha256=hashlib.sha256((run/'checkpoints/best.pt').read_bytes()).hexdigest(),
                scope='TRAIN crop diagnostic only; independent Gaussian noise, fixed 20 DDIM steps; not validation performance')
    write_json(run/'p0_gate.json',report)
    del model,state;torch.cuda.empty_cache()
    return report

def main():
    BATCH.mkdir(parents=True,exist_ok=True)
    lock=open(ROOT/'outputs/.experiments.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        status('running','preflight')
        plan=dict(batch_id=BATCH_ID,experiments=[dict(name=name,config=json.loads((ROOT/path).read_text()),run_id=BATCH_ID+'_'+name) for name,path in PLAN],
                  source_sha256=source_hashes(ROOT),scope='Two independent 600-update P0 ablations, same initialization seed, crops and gate. No automatic P1 or extra training.',
                  baseline='outputs/launches/diffusion_resume_20261017/p0_gate.json')
        plan_path=BATCH/'plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise RuntimeError('Frozen ablation plan/source changed; use a separately reviewed batch')
        write_json(plan_path,plan)
        command(ROOT/'scripts/check_environment.py','--output',BATCH/'environment.json')
        command('-m','unittest','discover','-s','tests','-v')
        command(ROOT/'scripts/verify_pipeline.py','--output',BATCH/'pipeline_verification.json')
        command(ROOT/'work/verify_high_noise_resume.py','--output',BATCH/'high_noise_resume.json')
        reports={}
        for name,path in PLAN:
            status('running','training_'+name)
            ident=BATCH_ID+'_'+name;run=ROOT/'outputs/P0/diffusion/seed_42'/ident
            expected=json.loads((ROOT/path).read_text())
            if run.exists():
                if json.loads((run/'config.json').read_text())!=expected:raise RuntimeError('Run config mismatch')
                if json.loads((run/'run_metadata.json').read_text())['source_sha256']!=source_hashes(ROOT):raise RuntimeError('Run source mismatch')
                if json.loads((run/'status.json').read_text())['status']!='completed':command(ROOT/'scripts/train.py','--resume',run/'checkpoints/last.pt')
            else:command(ROOT/'scripts/train.py','--config',ROOT/path,'--run-id',ident)
            status('running','evaluating_'+name)
            reports[name]=assess(run)
            write_json(BATCH/'partial_results.json',reports)
        baseline=json.loads((ROOT/'outputs/launches/diffusion_resume_20261017/p0_gate.json').read_text())
        write_json(BATCH/'summary.json',dict(ablations=reports,baseline_diffusion=[x for x in baseline['results'] if x['method']=='diffusion'],
                   scope=plan['scope'],source_sha256=source_hashes(ROOT)))
        status('completed','finished',gate_passed={key:value['passed'] for key,value in reports.items()})
    except BaseException as exc:
        status('failed','error',error=str(exc));traceback.print_exc();raise

if __name__=='__main__':main()

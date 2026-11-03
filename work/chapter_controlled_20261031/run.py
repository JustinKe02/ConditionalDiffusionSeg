"""Durable, locked, serial execution of the approved I/N/F study."""
import os,sys,json,time,fcntl,subprocess,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
import core as C
G=C.G
HERE=Path(__file__).resolve().parent
PLAN=C.BATCH/'execution_plan.json'

def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

def status(**kw):
    p=C.BATCH/'status.json'; old=json.loads(p.read_text()) if p.exists() else {}
    old.update(kw,updated_bjt=now()); G.atomic_json(p,old)

def guard(data=False):
    p=json.loads(PLAN.read_text())
    for rel,h in p['frozen_sha256'].items(): assert G.digest(C.ROOT/rel)==h,'Frozen source changed: '+rel
    if data:
        for path,h in json.loads((C.BATCH/'train_val_data_sha256.json').read_text()).items(): assert G.digest(path)==h,'Data changed: '+path

def snapshot_data():
    out={}; manifest=json.loads((C.ROOT/'data_manifests/source_dataset.json').read_text()); root=Path(manifest['dataset_root'])
    for split in ['train','val']:
        for ident in Path(manifest['splits'][split]['snapshot_path']).read_text().split():
            for pattern in ['image_pattern','mask_pattern']:
                path=root/manifest[pattern].format(id=ident); out[str(path)]=G.digest(path)
    G.atomic_json(C.BATCH/'train_val_data_sha256.json',out)

def freeze():
    proposal=json.loads(C.PROPOSAL.read_text())
    for rel,h in proposal['source_sha256'].items(): assert G.digest(C.ROOT/rel)==h,rel
    assert json.loads((C.BATCH/'engineering_verification.json').read_text())['passed']
    ready=json.loads((C.BATCH/'implementation_ready.json').read_text())
    for name,h in ready['source_sha256'].items(): assert G.digest(HERE/name)==h,name
    files=set(proposal['source_sha256'])
    files.update(str(p.relative_to(C.ROOT)) for p in HERE.glob('*.py'))
    files.update(str(p.relative_to(C.ROOT)) for p in (C.ROOT/'src/conditional_diffusion').glob('*.py'))
    files.update(str(p.relative_to(C.ROOT)) for p in [C.PROPOSAL,C.BATCH/'authorization.json',C.BATCH/'train_val_data_sha256.json',C.BATCH/'engineering_verification.json',C.BATCH/'implementation_ready.json'])
    ids=Path(json.loads((C.ROOT/'data_manifests/source_dataset.json').read_text())['splits']['val']['snapshot_path']).read_text().split()
    G.atomic_json(PLAN,dict(status='approved_frozen_execution',created_bjt=now(),authorization=str(C.BATCH/'authorization.json'),
                   proposal=str(C.PROPOSAL),order=['I','N','F'],configs={a:C.config(a) for a in ['I','N','F']},runs={a:str(C.RUNS[a]) for a in ['I','N','F']},
                   benchmark_warmup_image=ids[0],benchmark_timed_ids=ids[1:],frozen_sha256={rel:G.digest(C.ROOT/rel) for rel in sorted(files)},
                   new_test_inference=False,scope='Approved three100ep seed42 train/val arms; referenceR reused; no adaptive training'))

def call(script,args,stage):
    status(stage=stage,stage_started_bjt=now(),error=None)
    with (C.BATCH/f'{stage}.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u','-B',str(HERE/script),*args],stdout=log,stderr=subprocess.STDOUT)
        status(child_pid=child.pid)
        result=child.wait()
    status(child_pid=None)
    if result: raise RuntimeError(f'{stage} exited {result}; see {stage}.log')

def coordinator():
    C.BATCH.mkdir(parents=True,exist_ok=True)
    lock=open(C.ROOT/'outputs/.experiments.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    auth=json.loads((C.BATCH/'authorization.json').read_text()); assert auth['approved']
    status(stage='initializing',coordinator_pid=os.getpid(),completed=False,started_bjt=now())
    try:
        if not PLAN.exists():
            if not (C.BATCH/'train_val_data_sha256.json').exists(): snapshot_data()
            if not (C.BATCH/'engineering_verification.json').exists(): call('verify.py',[],'engineering_verification')
            status(stage='waiting_implementation_ready')
            while not (C.BATCH/'implementation_ready.json').exists(): time.sleep(10)
            freeze()
        guard(data=True)
        for a in ['I','N','F']:
            st=C.RUNS[a]/'status.json'
            if st.exists() and json.loads(st.read_text()).get('status')=='completed': continue
            guard(data=True); call('run.py',['--worker',a],'training_'+a)
        for a in ['R','I','N','F']:
            if not (C.ANALYSIS/a/'analysis_complete.json').exists():
                guard(data=True); call('analysis.py',['--arm',a],'analysis_'+a)
        for a in ['R','I','N','F']:
            for k in [1,2,4]:
                if not (C.ANALYSIS/'benchmark'/f'{a}_stop{k}.json').exists():
                    guard(); call('benchmark.py',['--arm',a,'--stop',str(k)],f'benchmark_{a}_{k}')
        guard(data=True); call('analysis.py',['--report'],'final_report'); guard(data=True)
        status(stage='completed',completed=True,finished_bjt=now(),child_pid=None)
    except BaseException as e:
        status(stage='failed',error=str(e),traceback=traceback.format_exc(),child_pid=None)
        raise
    finally: lock.close()

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        guard(data=True); a=sys.argv[2]; p=json.loads(PLAN.read_text())
        C.train(p['configs'][a],p['runs'][a],G.digest(PLAN),guard_fn=lambda:guard())
    else: coordinator()

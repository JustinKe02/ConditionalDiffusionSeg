"""Resume benchmark and reporting without changing any frozen project source."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, hashlib, fcntl, shutil
from pathlib import Path
from datetime import datetime
ROOT = Path('/root/autodl-tmp/ConditionalDiffusionSeg')
sys.path[:0] = [str(ROOT/'work'), str(ROOT/'src')]
import torch
import benchmark_ch4_20261024 as B
import launch_ch4_extension_20261024 as L

class GlobalOnlyTracker:
    """Count the same registered ATen FLOPs without module/autograd hooks."""
    parents = {'Global'}
    def __enter__(self): return self
    def __exit__(self, *args): return False

def flops_of(fn):
    with torch.utils.flop_counter.FlopCounterMode(display=False) as counter:
        counter.mod_tracker.__exit__()
        counter.mod_tracker = GlobalOnlyTracker()
        fn()
    return int(counter.get_total_flops())

def verify_counter():
    with torch.no_grad():
        conv=torch.nn.Conv2d(3,5,3,bias=False)
        x=torch.randn(2,3,11,13)
        analytic=2*2*5*9*11*3*3*3
        assert flops_of(lambda:conv(x)) == analytic
        a,b=torch.randn(2,3,4),torch.randn(2,4,5)
        assert flops_of(lambda:torch.bmm(a,b)) == 2*2*3*4*5
        leaf=torch.nn.Parameter(torch.randn(4,8))
        norm=torch.nn.LayerNorm(8)
        model=lambda: norm(leaf)
        assert torch.equal(model(),model())
        assert flops_of(model)==0
    return {'conv_analytic':True,'bmm_analytic':True,'no_grad_parameter_layernorm':True,
            'counting':'unchanged PyTorch ATen flop registry, Global total only; module autograd hooks disabled'}

B.flops_of=flops_of
B.__file__=str(Path(__file__).resolve())
if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--one':
        B.one(sys.argv[2])
    else:
        lock=open(ROOT/'outputs/.experiments.lock','a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=json.loads(L.PLAN.read_text())
        L.guard_stage(plan)
        assert json.loads(L.EXPORT_REPORT.read_text())['all_checks_passed']
        assert json.loads(L.VERIFY_REPORT.read_text())['passed']
        assert json.loads((Path(plan['repela_run'])/'status.json').read_text())['status']=='completed'
        assert (L.R.TEST_OUT/'test_metrics.json').exists()
        rev=L.R.BATCH/'completion_revision_20261024'
        rev.mkdir(exist_ok=True)
        old=L.R.BATCH/'status.json'
        if not (rev/'status_before.json').exists(): shutil.copy2(old,rev/'status_before.json')
        checks=verify_counter()
        manifest={'started_bjt':L.now(),'original_plan_sha256':L.G.digest(L.PLAN),
                  'runner_sha256':L.G.digest(Path(__file__)), 'counter_checks':checks,
                  'reused_objects':[p.stem for p in sorted((B.OUT/'objects').glob('*.json'))],
                  'scope':'benchmark remaining objects and report only; no training, test, source/data edits'}
        L.G.atomic_json(rev/'revision.json',manifest)
        L.status(stage='benchmark_resuming',pid=os.getpid(),revision=str(rev))
        try:
            B.main()
            L.guard_stage(plan)
            L.report(plan)
            L.guard_stage(plan)
            L.status(stage='completed',finished_bjt=L.now(),returncode=0)
            manifest.update(finished_bjt=L.now(),completed=True)
            L.G.atomic_json(rev/'revision.json',manifest)
            print('COMPLETION VERIFIED',flush=True)
        except BaseException as exc:
            L.status(stage='failed_benchmark_revision',error=repr(exc))
            raise

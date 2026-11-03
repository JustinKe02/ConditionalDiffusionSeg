"""Complete analysis-path checks before issuing the coordinator's readiness flag."""
import sys,json,time,traceback
from pathlib import Path
import numpy as np
import torch
import core as C
import analysis as A
G=C.G

def synthetic_report_check():
    real_out,real_runs=C.ANALYSIS,C.RUNS
    fixture=C.BATCH/'preflight/synthetic_report_test'
    try:
        C.ANALYSIS=fixture; C.RUNS={a:fixture/a/'fake_run' for a in ['R','I','N','F']}
        for a in C.RUNS:
            folder=fixture/a; folder.mkdir(parents=True,exist_ok=True); C.RUNS[a].mkdir(parents=True,exist_ok=True)
            G.atomic_json(folder/'analysis_complete.json',{'meta':{'epoch':100},'scope':'SYNTHETIC TEST ONLY'})
            G.atomic_json(folder/'trajectory_metrics.json',{'scope':'SYNTHETIC TEST ONLY'})
            G.atomic_json(C.RUNS[a]/'validation_epoch100.json',{'miou_4class':1.})
            cm=np.repeat((np.eye(4,dtype=np.int64)*100)[None],25,axis=0)
            bc=np.tile(np.array([10,12,10,11]),(25,4,1))
            np.savez_compressed(folder/'trajectory_counts.npz',ids=np.array(['synthetic'+str(i) for i in range(25)]),s4_cm=cm,s4_bf2=bc)
            for k in [1,2,4]: G.atomic_json(fixture/'benchmark'/f'{a}_stop{k}.json',{'image_predict_s_median':1.,'synthetic':True})
        A.report()
        r=json.loads((fixture/'comparison_summary.json').read_text())
        assert all(v['differences']['miou']['point_pp']==0 for v in r['contrasts'].values())
    finally: C.ANALYSIS,C.RUNS=real_out,real_runs

def main():
    while True:
        p=C.BATCH/'status.json'; st=json.loads(p.read_text()) if p.exists() else {}
        if st.get('stage')=='failed': raise RuntimeError(st.get('error'))
        if st.get('stage')=='waiting_implementation_ready': break
        time.sleep(5)
    report={'started_unix':time.time(),'checks':{}}
    A.metric_selfcheck(); report['checks']['cpu_metric_identities']=True
    model,c,meta=C.load_best('R'); d=G.GLPDataset(c,'val',train=False)[0]
    endpoints,nfe=C.predict(model,c,d['image'],d['thumb'],d['id'],capture=(1,2,4))
    for k in [1,2,4]:
        p,calls=C.predict(model,c,d['image'],d['thumb'],d['id'],stop_after=k)
        assert np.array_equal(p,endpoints[k]),k
        assert calls==35*k
    report['checks']['fixed_K4_early_return_matches_captured_endpoints']={'id':d['id'],'steps':[1,2,4],'exact':True}
    gt=d['mask'].numpy()
    for tol in [1,2,4]:
        assert np.array_equal(A.boundary_counts(gt,endpoints[4],tol),G.boundary_counts(gt,endpoints[4],tol))
        bi=A.boundary_iou_counts(gt,endpoints[4],tol)
        assert bool((bi[:,0]<=bi[:,1]).all())
    report['checks']['native_image_boundary_metrics']=True
    for intervention in ['zero_state','zero_selfcond']:
        p,_=C.predict(model,c,d['image'],d['thumb'],d['id'],intervention=intervention)
        assert p.shape==gt.shape and p.min()>=0 and p.max()<4
    report['checks']['both_intervention_paths']=True
    torch.use_deterministic_algorithms(False); torch.backends.cudnn.deterministic=False; torch.backends.cudnn.benchmark=True
    p,_=C.predict(model,c,d['image'],d['thumb'],d['id'],stop_after=2)
    agreement=float((p==endpoints[2]).mean()); assert agreement>=.9999
    timing={}; C.predict(model,c,d['image'],d['thumb'],d['id'],stop_after=2,timing=timing)
    assert timing['network_seconds']>0
    report['checks']['native_benchmark_path']={'agreement':agreement,'network_seconds':timing['network_seconds']}
    synthetic_report_check(); report['checks']['synthetic_report_and_bootstrap']=True
    report.update(passed=True,finished_unix=time.time())
    G.atomic_json(C.BATCH/'analysis_engineering_verification.json',report)
    sources={p.name:G.digest(p) for p in Path(__file__).parent.glob('*.py')}
    G.atomic_json(C.BATCH/'implementation_ready.json',dict(ready=True,created_unix=time.time(),source_sha256=sources,
                  analysis_engineering_report_sha256=G.digest(C.BATCH/'analysis_engineering_verification.json')))
    print('IMPLEMENTATION READY; approved coordinator may continue automatically',flush=True)

if __name__=='__main__':
    try: main()
    except BaseException as e:
        G.atomic_json(C.BATCH/'analysis_engineering_failed.json',dict(error=str(e),traceback=traceback.format_exc(),time_unix=time.time()))
        raise

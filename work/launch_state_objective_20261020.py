"""Exactly P0/P1 scratch40, fixed-best sampling checks, report and stop."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import fcntl
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'work'))
from conditional_diffusion.runtime import write_json,source_hashes
from conditional_diffusion.data import read_manifest
from conditional_diffusion.metrics_impl import summarize,summarize_boundaries
from state_objective_worker import make_config

BATCH=ROOT/'outputs/launches/p1_state_objective_20261020'
PREVIOUS=ROOT/'outputs/launches/p1_static_semantic_20261020'
DIAG=ROOT/'outputs/analysis/static_state_diagnostic_20261020'
STAGE='preflight'
MAPS={'P0':'tanh','P1':'simplex'}
SCREEN=dict(primary_P1_minus_S0_minimum_miou=.005,primary_P1_minus_S1_minimum_miou=.003,
    primary_P1_minus_P0_minimum_miou=.002,mean_mono_few_iou_gain_vs_S0_minimum=.005,
    maximum_thin_class_or_foreground_boundary_decline_vs_S0=.005,
    secondary_all_P1_miou_differences_strictly_positive=True,
    scope='Exploratory consistency screen, single trainingseed42, not significance or established novelty')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(value,**extra):
    record=dict(status=value,stage=STAGE,pid=os.getpid(),updated_at_utc=datetime.now(timezone.utc).isoformat(),**extra)
    write_json(BATCH/'status.json',record);print(record,flush=True)


def history_check():
    for path,digest in read(BATCH/'source_before.json')['protected_history_sha256'].items():
        assert sha(ROOT/path)==digest,'Historical frozen file mismatch: '+path


def verify(plan):
    history_check()
    assert source_hashes(ROOT)==plan['source_sha256'],'Current source/config mismatch'
    for path,digest in plan['workflow_sha256'].items():
        assert sha(ROOT/path)==digest,'Workflow/preflight mismatch: '+path
    read_manifest(plan['configs']['P1']['manifest'])


def worker(action,variant,path,plan=False):
    command=[sys.executable,'-u',str(ROOT/'work/state_objective_worker.py'),action,'--variant',variant,'--output',str(path)]
    if plan:command+=['--plan',str(BATCH/'plan.json')]
    with path.with_suffix('.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=read(path)
    assert result['status'] in ['passed','completed'],result
    return result


def prepare():
    global STAGE
    if (BATCH/'plan.json').exists():
        plan=read(BATCH/'plan.json');verify(plan);return plan
    history_check()
    diagnosis=read(DIAG/'summary.json')
    assert diagnosis['status']=='completed' and diagnosis['optimizer_steps']==0 and not diagnosis['optimizer_constructed']
    cpu=read(BATCH/'cpu_verification.json');assert cpu['status']=='passed'
    assert cpu['tests_run']==68 and all(row['parameter_count']==5939156
        and row['initial_state_exactly_equals_historical_S2'] and row['additional_learned_parameters']==0
        for row in cpu['models'].values())
    cuda=read(BATCH/'cuda_verification.json');assert cuda['status']=='passed'
    for path,digest in cpu['source_sha256'].items():assert sha(ROOT/path)==digest,path
    assert cuda['source_sha256']==source_hashes(ROOT),'CUDA preflight code/config mismatch'
    for variant in MAPS:
        for action,prefix in [('benchmark','benchmark_train'),('window_benchmark','benchmark_window')]:
            STAGE=f'{action}_{variant}';status('running')
            output=BATCH/f'{prefix}_{variant}.json'
            row=read(output) if output.exists() else worker(action,variant,output)
            assert row['status']=='passed'
            if action=='benchmark':
                assert row['warmup_updates']==10 and row['timed_updates']==30 and row['batch']==2
                assert row['peak_reserved_mib']<=.9*row['total_memory_mib']
            else:assert row['window_batch_size']==1 and row['nfe']==175
    STAGE='freezing';status('running')
    configs={v:make_config(m) for v,m in MAPS.items()}
    differences={k:[configs['P0'].get(k),configs['P1'].get(k)] for k in set(configs['P0'])|set(configs['P1'])
                 if configs['P0'].get(k)!=configs['P1'].get(k)}
    assert set(differences)=={'experiment_variant','state_mapping'}
    for variant,c in configs.items():
        assert c['seed']==42 and c['epochs']==40 and c['batch_size']==2 and c['window_batch_size']==1
        assert c['mode']=='S2' and c['sampling_steps']==5 and c['sampling_seed']==3407
        assert c['sampler_final_readout']=='raw_logits' and not c['class_balance'] and not c['boundary']
        path=ROOT/f'configs/pilot_state_objective_{variant.lower()}.json'
        if path.exists():assert read(path)==c
        else:write_json(path,c)
    previous=read(PREVIOUS/'comparison_summary.json')['results']
    baselines={v:previous[v] for v in ['S0','S1','S2','S2_secondary']}
    paths=[ROOT/'work/launch_state_objective_20261020.py',ROOT/'work/state_objective_worker.py',
           ROOT/'work/verify_state_objective_20261020.py',ROOT/'work/verify_state_objective_cuda_20261020.py',
           ROOT/'tests/test_state_objective.py',ROOT/'work/diagnose_static_state_20261020.py']
    paths+=list(DIAG.glob('*.json'))+[DIAG/'README.md']
    paths+=[BATCH/x for x in ['source_before.json','authorization.json','cpu_verification.json','cuda_verification.json',
            'benchmark_train_P0.json','benchmark_train_P1.json','benchmark_window_P0.json','benchmark_window_P1.json',
            'README.md','research_rationale.json','unit_tests.log']]
    manifest=read_manifest(configs['P1']['manifest'])
    paths+=[Path(configs['P1']['manifest'])]+[Path(row['snapshot_path']) for row in manifest['splits'].values()]
    for row in cuda['results'].values():
        path=Path(row['checkpoint']);assert sha(path)==row['checkpoint_sha256'];paths.append(path)
    source=source_hashes(ROOT)
    added=sorted(set(source)-set(read(BATCH/'source_before.json')['source_before_sha256']))
    assert added==sorted(['src/conditional_diffusion/state_objective.py',
        'configs/pilot_state_objective_p0.json','configs/pilot_state_objective_p1.json']),added
    plan=dict(batch_id=BATCH.name,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        authorization=read(BATCH/'authorization.json'),configs=configs,config_differences=differences,
        runs={v:str(ROOT/f'outputs/P1/state_objective/seed_42/{BATCH.name}_{v.lower()}') for v in MAPS},
        source_sha256=source,source_change_set=added,workflow_sha256={str(path.relative_to(ROOT)):sha(path) for path in paths},
        baselines=baselines,screening=SCREEN,parameters=dict(P0=5939156,P1=5939156),
        budget=dict(new_training_count=2,training_seed=42,epochs=40,batch=2,updates_per_run=2400,
                    records_per_epoch=120,unique_rgb=119,exposures_per_run=4800,initialization='scratch; benchmark weights discarded'),
        bindings='Dedicated worker explicitly binds shared.build=build_state_model and shared.sample_cached=sample_scores; frozen old loop/loss/recycling/inference are unchanged.',
        hypothesis='Bounded x0 maps may reduce raw-logit regression/classification conflict; P0 controls generic boundedness, P1 adds mutually exclusive class-probability expectation.',
        interpretation='Both new arms use raw semantic final readout; old S2 used clipped state readout. New-vs-old S2 is a joint parameterization/readout change, not a pure loss-only attribution. P1-vs-P0 keeps this readout fixed.',
        batch_choice='Reuse completed all-mode2/4/8/16 benchmark: S1 full-five-step backward only stable at2, preserving matched S0/S1 controls. New2-state mappings benchmarked realB2 10+30; windowbatch1 fixed.',
        scope='Exactly P0 then P1 scratch40, full25val every10 atfiveDDIM3407 selectsbest, eachsamebest3408, compare/report/stop. MoS2train/valonly; no othermaterials/test/formal200/modulequeue.')
    write_json(BATCH/'plan.json',plan);verify(plan);return plan


def metrics_audit(path,metrics):
    with np.load(path) as data:
        assert len(data['ids'])==25 and np.array_equal(data['per_image'].sum(0),data['global_matrix'])
        recomputed=dict(**summarize(data['global_matrix']),**summarize_boundaries(data['boundary_counts']))
        assert all(abs(value-metrics[k])<1e-12 for k,value in recomputed.items())


def secondary(variant,plan):
    run=Path(plan['runs'][variant]);checkpoint=run/'checkpoints/best.pt';digest=sha(checkpoint)
    expected=read(run/'best_val_metrics.json')['epoch'];path=BATCH/f'{variant}_seed3408.json';record=None
    if path.exists():
        try:record=read(path)
        except json.JSONDecodeError:pass
        if not (isinstance(record,dict) and record.get('status')=='completed' and record.get('epoch')==expected
                and record.get('sampling_seed')==3408 and record.get('checkpoint_sha256')==digest):
            stamp=str(time.time_ns())
            for artifact in [path,path.with_suffix('.npz'),path.with_name(path.stem+'_timing.json')]:
                if artifact.exists():artifact.rename(artifact.with_name(artifact.stem+'_failed_'+stamp+artifact.suffix))
            record=None
    if record is None:
        record=worker('secondary',variant,path,True);assert record['epoch']==expected and sha(checkpoint)==digest
        record['checkpoint_sha256']=digest;write_json(path,record)
    metrics_audit(path.with_suffix('.npz'),record['metrics']);verify(plan);return record


def finish(plan,results):
    base={v:plan['baselines'][v]['metrics'] for v in ['S0','S1','S2']}
    p1,p0=results['P1']['3407']['metrics'],results['P0']['3407']['metrics']
    second=results['P1']['3408']['metrics']
    checks=dict(primary_vs_S0=p1['miou_4class']-base['S0']['miou_4class']>=.005,
        primary_vs_S1=p1['miou_4class']-base['S1']['miou_4class']>=.003,
        primary_vs_P0=p1['miou_4class']-p0['miou_4class']>=.002,
        thin_mean_gain=(p1['iou_monolayer']+p1['iou_fewlayer']-base['S0']['iou_monolayer']-base['S0']['iou_fewlayer'])/2>=.005,
        thin_boundary_guard=all(p1[k]-base['S0'][k]>=-.005 for k in ['iou_monolayer','iou_fewlayer','boundary_f1_foreground']),
        secondary_positive=all(second['miou_4class']>row['miou_4class'] for row in [base['S0'],base['S1'],results['P0']['3408']['metrics']]))
    summary=dict(results=results,baselines=plan['baselines'],screening=SCREEN,screening_checks=checks,
                 screening_passed=all(checks.values()),budget=plan['budget'],source_sha256=plan['source_sha256'],
                 interpretation=plan['interpretation'],stop='All planned work finished; no automatic rescue, new module, seed, budget, test or material queue.')
    write_json(BATCH/'comparison_summary.json',summary)
    lines=['# 有界状态与类别概率一致性筛选','', '| 模式 | best轮 | 主设置mIoU % | 同best3408 % | 单层IoU % | 少层IoU % | 前景边界F1 % |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in [('S0',plan['baselines']['S0']),('S1',plan['baselines']['S1']),('S2',plan['baselines']['S2'])]:
        m=row['metrics'];lines.append(f"| {name} | {row['epoch']} | {100*m['miou_4class']:.4f} | — | {100*m['iou_monolayer']:.4f} | {100*m['iou_fewlayer']:.4f} | {100*m['boundary_f1_foreground']:.4f} |")
    for name in MAPS:
        row=results[name]['3407'];m=row['metrics'];second=results[name]['3408']['metrics']
        lines.append(f"| {name} | {row['epoch']} | {100*m['miou_4class']:.4f} | {100*second['miou_4class']:.4f} | {100*m['iou_monolayer']:.4f} | {100*m['iou_fewlayer']:.4f} | {100*m['boundary_f1_foreground']:.4f} |")
    lines+=['','探索筛选：'+('通过' if all(checks.values()) else '未通过')+'。',
            '两组均为连续高斯扩散；单头、同容量、同最终raw logits读出。类别概率映射不直接作为已成立原创。',
            '3408只复核同best，不重选，不是独立训练。旧test未使用。全部结束，停止本批。']
    (BATCH/'final_analysis.md').write_text('\n'.join(lines)+'\n')


def main():
    global STAGE
    lock=(ROOT/'outputs/.experiments.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        plan=prepare();results={}
        for variant in MAPS:
            STAGE='training_'+variant;status('running',budget=plan['budget']);verify(plan)
            run=Path(plan['runs'][variant]);current={}
            try:current=read(run/'status.json')
            except (FileNotFoundError,json.JSONDecodeError):pass
            if current.get('status')!='completed':worker('train',variant,BATCH/f'train_{variant}_result.json',True)
            assert read(run/'status.json')==dict(status='completed',last_completed_epoch=40,global_step=2400)
            row=read(run/'best_val_metrics.json');assert row['metrics']['image_count']==25 and row['metrics']['nfe_total']==4375
            metrics_audit(run/'best_val_confusion.npz',row['metrics']);verify(plan)
            STAGE=variant+'_samebest3408';status('running')
            results[variant]={'3407':dict(**row,checkpoint_sha256=sha(run/'checkpoints/best.pt')),'3408':secondary(variant,plan)}
            write_json(BATCH/'partial_results.json',results)
        STAGE='reporting';status('running');finish(plan,results);verify(plan)
        STAGE='finished';status('completed',screening_passed=read(BATCH/'comparison_summary.json')['screening_passed'])
    except BaseException as exc:
        status('failed',error=str(exc));traceback.print_exc();raise


if __name__=='__main__':main()

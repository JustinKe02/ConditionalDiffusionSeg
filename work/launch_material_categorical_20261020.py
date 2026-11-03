"""Bounded N0→N1→N2→N3 new100epoch batch; preflight, freeze, report, stop."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import fcntl
import json
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'work'))
from conditional_diffusion.material_conditions import digest,load_direct
from conditional_diffusion.runtime import write_json,source_hashes
from conditional_diffusion.data import read_manifest
from conditional_diffusion.metrics_impl import summarize,summarize_boundaries
from material_categorical_worker import make_config

BATCH=ROOT/'outputs/launches/p1_material_categorical_20261020'
MODES=['N0','N1','N2','N3'];STAGE='preflight'
SCREEN=dict(N3_minus_N0_minimum_miou=.01,N3_minus_N1_minimum_miou=.005,N3_minus_N2_minimum_miou=.005,
            mean_mono_few_gain_vs_N0_minimum=.01,maximum_individual_thin_boundary_decline=.005,
            samebest3408_positive_vs_N0_N1_N2=True,scope='singletrainingseed42 exploratory research-value screen, not significance/novelty')


def read(path):return json.loads(Path(path).read_text())
def status(value,**extra):
    row=dict(status=value,stage=STAGE,pid=os.getpid(),updated_at_utc=datetime.now(timezone.utc).isoformat(),**extra)
    write_json(BATCH/'status.json',row);print(row,flush=True)


def history_check():
    for p,h in read(BATCH/'source_before.json')['protected_history_sha256'].items():assert digest(ROOT/p)==h,'Protected history changed: '+p


def cache_check():
    index=read(BATCH/'conditions/index.json');assert index['status']=='completed' and index['image_count']==145
    assert digest(BATCH/'conditions/optical_statistics.json')==index['statistics_sha256']
    manifest=read_manifest(ROOT/'data_manifests/source_dataset.json');root=Path(manifest['dataset_root'])
    for ident,row in index['images'].items():
        assert digest(BATCH/'conditions'/(ident+'.npy'))==row['prior_file_sha256'],ident
        assert digest(root/manifest['image_pattern'].format(id=ident))==row['image_sha256'],ident
        if row['split']=='train':assert digest(root/manifest['mask_pattern'].format(id=ident))==row['train_mask_sha256'],ident
    return index


def verify(plan,full=False):
    history_check();assert source_hashes(ROOT)==plan['source_sha256'],'Source/config mismatch'
    for p,h in plan['workflow_sha256'].items():assert digest(ROOT/p)==h,'Workflow changed: '+p
    for p,h in plan['condition_metadata_sha256'].items():assert digest(ROOT/p)==h,p
    assert digest(plan['configs']['N0']['prior_checkpoint'])==plan['prior_sha256']
    if full:cache_check()


def worker(action,mode,out,batch=2,window=1,frozen=False):
    cmd=[sys.executable,'-u',str(ROOT/'work/material_categorical_worker.py'),action,'--variant',mode,
         '--batch',str(batch),'--window-batch',str(window),'--output',str(out)]
    if frozen:cmd+=['--plan',str(BATCH/'plan.json')]
    with out.with_suffix('.log').open('a') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return read(out)


def engineering_report(path):
    row=read(path);assert row['status']=='passed',str(path)
    for p,h in row['source_sha256'].items():assert digest(ROOT/p)==h,'Preflight source changed: '+p
    if 'worker_sha256' in row:assert digest(ROOT/'work/material_categorical_worker.py')==row['worker_sha256']
    return row


def prepare():
    global STAGE
    if (BATCH/'plan.json').exists():plan=read(BATCH/'plan.json');verify(plan,True);return plan
    history_check();cache=cache_check();cpu=engineering_report(BATCH/'cpu_verification.json');assert cpu['tests_run']==8
    cuda=engineering_report(BATCH/'cuda_verification.json');assert all(x['restore_exact'] for x in cuda['rows'].values())
    train_rows={};window_rows={}
    for mode in MODES:
        for batch in [2,4,8]:
            STAGE=f'train_throughput_{mode}_b{batch}';status('running')
            path=BATCH/f'benchmark_train_{mode}_b{batch}.json'
            row=read(path) if path.exists() else worker('benchmark_train',mode,path,batch)
            assert row['status'] in ['passed','oom']
            if row['status']=='passed':
                engineering_report(path);assert row['warmup_updates']==10 and row['timed_updates']==30
            train_rows[(mode,batch)]=row
    eligible=[b for b in [2,4,8] if all(train_rows[(m,b)]['status']=='passed' and
               train_rows[(m,b)]['peak_reserved_mib']<=.9*train_rows[(m,b)]['total_memory_mib'] for m in MODES)]
    assert eligible,'No safe common real training batch'
    scores={b:sum(math.log(train_rows[(m,b)]['samples_per_second']) for m in MODES)/4 for b in eligible}
    chosen=max(eligible,key=lambda b:scores[b]);assert chosen==2,'Selected batch differs from actualB2 restoration check; verify selected batch before freezing'
    for mode in MODES:
        for window in [1,4,8,16]:
            STAGE=f'window_throughput_{mode}_b{window}';status('running')
            path=BATCH/f'benchmark_window_{mode}_b{window}.json'
            row=read(path) if path.exists() else worker('benchmark_window',mode,path,chosen,window)
            assert row['status'] in ['passed','oom']
            if row['status']=='passed':engineering_report(path)
            window_rows[(mode,window)]=row
    windows=[b for b in [1,4,8,16] if all(window_rows[(m,b)]['status']=='passed' and
               window_rows[(m,b)]['peak_reserved_mib']<=.9*window_rows[(m,b)]['total_memory_mib'] for m in MODES)]
    assert windows,'No safe common window batch'
    best_window=min(windows,key=lambda b:sum(math.log(window_rows[(m,b)]['new_model_native_seconds']) for m in MODES))
    for mode in MODES:
        STAGE=f'cold_pipeline_{mode}';status('running');path=BATCH/f'benchmark_cold_{mode}.json'
        row=read(path) if path.exists() else worker('benchmark_cold',mode,path,chosen,best_window)
        assert row['status']=='passed';engineering_report(path)
    STAGE='freezing';status('running')
    configs={m:make_config(m,chosen,best_window) for m in MODES}
    for mode,c in configs.items():
        path=ROOT/f'configs/material_categorical_{mode.lower()}.json'
        if path.exists():assert read(path)==c
        else:write_json(path,c)
    old=read(BATCH/'source_before.json')['source_before_sha256'];source=source_hashes(ROOT)
    additions=sorted(set(source)-set(old));expected=[f'src/conditional_diffusion/{s}.py' for s in ['material_conditions','material_categorical','material_experiment']]+[f'configs/material_categorical_{m.lower()}.json' for m in MODES]
    assert additions==sorted(expected),additions
    for p,h in old.items():assert source[p]==h,p
    manifests=read_manifest(configs['N0']['manifest'])
    workflow=[ROOT/'work'/s for s in ['prepare_material_conditions_20261020.py','verify_material_categorical_20261020.py','material_categorical_worker.py','launch_material_categorical_20261020.py']]
    workflow+=[BATCH/s for s in ['source_before.json','authorization.json','README.md','cpu_verification.json','cpu_verification.log','cuda_verification.json']]
    workflow+=list(BATCH.glob('benchmark_*.json'))
    workflow+=[Path(configs['N0']['manifest'])]+[Path(x['snapshot_path']) for x in manifests['splits'].values()]
    for row in cuda['rows'].values():
        assert digest(row['checkpoint'])==row['checkpoint_sha256'];workflow.append(Path(row['checkpoint']))
    timings=dict(training=[dict(mode=m,batch=b,**{k:v for k,v in train_rows[(m,b)].items() if k not in ['mode','batch','source_sha256']}) for m in MODES for b in [2,4,8]],
                 windows=[dict(mode=m,window=b,**{k:v for k,v in window_rows[(m,b)].items() if k not in ['mode','window','source_sha256']}) for m in MODES for b in [1,4,8,16]],
                 selected_train_batch=chosen,selected_window_batch=best_window,
                 selection='max geometric mean train samples/s and min geometric mean native window latency among all-mode passes with10% reserve')
    write_json(BATCH/'throughput_summary.json',timings);workflow.append(BATCH/'throughput_summary.json')
    prior=load_direct(configs['N0']['prior_checkpoint']);prior_parameters=sum(p.numel() for p in prior.parameters());del prior
    plan=dict(batch_id=BATCH.name,frozen_at_utc=datetime.now(timezone.utc).isoformat(),authorization=read(BATCH/'authorization.json'),
              configs=configs,runs={m:str(ROOT/f'outputs/P1/material_categorical/seed_42/{BATCH.name}_{m.lower()}') for m in MODES},
              source_sha256=source,source_additions=additions,workflow_sha256={str(p.relative_to(ROOT)):digest(p) for p in workflow},
              condition_metadata_sha256={str((BATCH/'conditions'/s).relative_to(ROOT)):digest(BATCH/'conditions'/s) for s in ['index.json','optical_statistics.json']},
              prior_sha256=cache['prior_sha256'],prior_parameters=prior_parameters,screening=SCREEN,
              parameters={m:cuda['rows'][m]['parameters'] for m in MODES},
              budget=dict(new_trainings=4,new_epochs_per_mode=100,training_seed=42,batch=chosen,updates_per_mode=100*math.ceil(120/chosen),
                          new_exposures_per_mode=12000,shared_prior_completed_epochs=40,initialization='same completedDirect compatibleweights+matchedseed42newlayers; not100scratch'),
              scope='ExactlyN0,N1,N2,N3new100,full25valevery20,mainfive3407,N2/N3samebest3408,reportstop.No autoN4/publicmethods/material/test/stepmatrix.',
              cache_accounting='5.3GiB fixed prior cache generated once; prior reference timing separated from current cached evaluation; real cold prior+condition+model pipeline measured separately before training',
              hypothesis='Image/location-dependent substrate-relative optical class relations regulate categorical corruption and recovery beyond global relation and matched deterministic refinement; prior art acknowledged.')
    write_json(BATCH/'plan.json',plan);verify(plan,True);return plan


def metrics_audit(path,metrics):
    with np.load(path) as z:
        assert len(z['ids'])==25 and np.array_equal(z['per_image'].sum(0),z['global_matrix'])
        values={**summarize(z['global_matrix']),**summarize_boundaries(z['boundary_counts'])}
        assert all(abs(v-metrics[k])<1e-12 for k,v in values.items())


def secondary(mode,plan):
    run=Path(plan['runs'][mode]);best=read(run/'best_val_metrics.json');sha=digest(run/'checkpoints/best.pt');path=BATCH/f'{mode}_seed3408.json'
    row=None
    if path.exists():
        try:row=read(path)
        except json.JSONDecodeError:pass
        if not (isinstance(row,dict) and row.get('status')=='completed' and row.get('checkpoint_sha256')==sha and row.get('epoch')==best['epoch']):
            stamp=str(time.time_ns())
            for p in [path,path.with_suffix('.npz'),path.with_name(path.stem+'_timing.json')]:
                if p.exists():p.rename(p.with_name(p.stem+'_incomplete_'+stamp+p.suffix))
            row=None
    if row is None:row=worker('secondary',mode,path,frozen=True)
    assert row['epoch']==best['epoch'] and row['sampling_seed']==3408 and digest(run/'checkpoints/best.pt')==sha==row['checkpoint_sha256']
    metrics_audit(path.with_suffix('.npz'),row['metrics']);return row


def finish(plan,results):
    m={k:v['3407']['metrics'] for k,v in results.items()};n=m['N3'];base=m['N0']
    checks={f'N3_vs_{key}':n['miou_4class']-m[key]['miou_4class']>=threshold for key,threshold in [('N0',.01),('N1',.005),('N2',.005)]}
    checks.update(thin_mean=(n['iou_monolayer']+n['iou_fewlayer']-base['iou_monolayer']-base['iou_fewlayer'])/2>=.01,
                  thin_boundary_guard=all(n[k]-base[k]>=-.005 for k in ['iou_monolayer','iou_fewlayer','boundary_f1_foreground']),
                  secondary_positive=all(results['N3']['3408']['metrics']['miou_4class']>x['miou_4class'] for x in [base,m['N1'],results['N2']['3408']['metrics']]))
    summary=dict(results=results,screening=SCREEN,screening_checks=checks,screening_passed=all(checks.values()),budget=plan['budget'],
                 stop='All authorized main work complete; no automatic extension/modules/othermaterials/test/conditionalfollowups.')
    write_json(BATCH/'comparison_summary.json',summary)
    lines=['# 材料条件类别扩散：主比较完成','', '| 模式 | best轮 | 主mIoU/% | 同best3408/% | 单层IoU/% | 少层IoU/% | 前景边界F1/% |','|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        row=results[mode]['3407'];v=row['metrics'];second=results[mode].get('3408');ss='—' if second is None else f"{100*second['metrics']['miou_4class']:.4f}"
        lines.append(f"| {mode} | {row['epoch']} | {100*v['miou_4class']:.4f} | {ss} | {100*v['iou_monolayer']:.4f} | {100*v['iou_fewlayer']:.4f} | {100*v['boundary_f1_foreground']:.4f} |")
    lines+=['','主筛选：'+('通过' if all(checks.values()) else '未通过')+'。',
            '四组共同继承Direct40轮并新增100轮；不是100轮从零训练。3408仅推理复核。',
            '验证seconds含缓存读取、融合和指标；先验已预计算，参考先验时间另列。完整冷启动推理测量见benchmark_cold，不混称当前评估时长。',
            'N2是机制对照，不是公开论文完整复现。全部完成后停止，无额外队列。']
    (BATCH/'final_analysis.md').write_text('\n'.join(lines)+'\n')


def main():
    global STAGE
    lock=(ROOT/'outputs/.experiments.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        plan=prepare();results={}
        for mode in MODES:
            STAGE='training_'+mode;status('running',budget=plan['budget']);verify(plan,True)
            run=Path(plan['runs'][mode]);current=read(run/'status.json') if (run/'status.json').exists() else {}
            if current.get('status')!='completed':worker('train',mode,BATCH/f'train_{mode}_result.json',frozen=True)
            assert read(run/'status.json')==dict(status='completed',last_completed_epoch=100,global_step=plan['budget']['updates_per_mode'])
            row=read(run/'best_val_metrics.json');metrics_audit(run/'best_val_confusion.npz',row['metrics'])
            results[mode]={'3407':dict(**row,checkpoint_sha256=digest(run/'checkpoints/best.pt'))}
            if mode in ['N2','N3']:
                STAGE=mode+'_samebest3408';status('running');verify(plan);results[mode]['3408']=secondary(mode,plan)
            write_json(BATCH/'partial_results.json',results)
        STAGE='reporting';status('running');finish(plan,results);verify(plan,True)
        STAGE='finished';status('completed',screening_passed=read(BATCH/'comparison_summary.json')['screening_passed'])
    except BaseException as exc:status('failed',error=str(exc));traceback.print_exc();raise


if __name__=='__main__':main()

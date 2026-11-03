"""Bounded authorized launch: benchmark, freeze, S0/S1/S2 (or exact S0 reuse), stop."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import fcntl
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'work'))
from conditional_diffusion.runtime import source_hashes, write_json
from conditional_diffusion.data import read_manifest
from conditional_diffusion.metrics_impl import summarize
from static_semantic_worker import base_config

BATCH = ROOT/'outputs/launches/p1_static_semantic_20261020'
HISTORICAL_DIRECT = ROOT/'outputs/P1/direct/seed_42/p1_cosine_20261017'
STAGE = 'initial_checks'
SCREEN = dict(s2_minus_s0_miou_minimum=.005, s2_minus_s1_miou_minimum=.003,
              s2_minus_s0_mean_monolayer_fewlayer_iou_minimum=.005,
              maximum_individual_thin_class_or_foreground_boundary_decline_vs_s0=.005,
              secondary_3408_both_miou_differences_strictly_positive=True,
              interpretation='exploratory screen; one training seed42; no significance or novelty established')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(value, **extra):
    write_json(BATCH/'status.json', dict(status=value, stage=STAGE, pid=os.getpid(),
               updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra))
    print(datetime.now(timezone.utc).isoformat(), STAGE, value, extra, flush=True)


def legacy_verify():
    before = read(BATCH/'source_before.json')
    for category, items in before['legacy_categories'].items():
        for path, digest in items.items():
            assert sha(ROOT/path) == digest, 'Legacy hash mismatch '+category+': '+path


def verify(plan):
    legacy_verify()
    assert source_hashes(ROOT) == plan['source_sha256'], 'New source/config hash mismatch'
    for path, digest in plan['workflow_sha256'].items():
        assert sha(ROOT/path) == digest, 'Workflow/preflight hash mismatch '+path
    read_manifest(plan['configs']['S2']['manifest'])


def worker(action, mode, batch, output, plan=None):
    cmd = [sys.executable, '-u', str(ROOT/'work/static_semantic_worker.py'), action,
           '--mode', mode, '--batch', str(batch), '--output', str(output)]
    if plan is not None:
        cmd += ['--plan', str(plan)]
    with output.with_suffix('.log').open('a') as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return read(output)


def benchmarks():
    global STAGE
    frozen = source_hashes(ROOT)
    identity = dict(source_sha256=frozen, worker_sha256=sha(ROOT/'work/static_semantic_worker.py'),
                    warmup_train_updates=10, timed_train_updates=30, train_batches=[2,4,8,16], window_batches=[1,4,8,16])
    if (BATCH/'benchmark_protocol.json').exists():
        assert read(BATCH/'benchmark_protocol.json') == identity
    else:
        write_json(BATCH/'benchmark_protocol.json', identity)
    result = {'train':{}, 'windows':{}}
    # S1 is usually the limiting memory path. All failures remain separate records.
    for mode in ['S1','S0','S2']:
        result['train'][mode] = {}
        for batch in [2,4,8,16]:
            STAGE = f'benchmark_train_{mode}_b{batch}'; status('running')
            assert source_hashes(ROOT) == frozen
            output = BATCH/f'bench_train_{mode}_b{batch}.json'
            row = read(output) if output.exists() else worker('benchmark_train', mode, batch, output)
            assert row['status'] in ['passed','oom'], row
            result['train'][mode][str(batch)] = row
    for mode in ['S0','S1','S2']:
        result['windows'][mode] = {}
        reference = None
        for batch in [1,4,8,16]:
            STAGE = f'benchmark_windows_{mode}_b{batch}'; status('running')
            assert source_hashes(ROOT) == frozen
            output = BATCH/f'bench_windows_{mode}_b{batch}.json'
            row = read(output) if output.exists() else worker('benchmark_windows', mode, batch, output)
            assert row['status'] in ['passed','oom'], row
            if row['status'] == 'passed':
                with np.load(output.with_suffix('.npz')) as data:
                    probability, prediction = data['probabilities'].copy(), data['prediction'].copy()
                if batch == 1:
                    reference = (probability, prediction)
                assert reference is not None
                row['max_abs_probability_error_vs_b1'] = float(np.max(np.abs(probability-reference[0])))
                row['changed_prediction_pixel_fraction_vs_b1'] = float(np.mean(prediction != reference[1]))
                row['numerical_guard_passed'] = row['max_abs_probability_error_vs_b1'] <= 1e-4 and row['changed_prediction_pixel_fraction_vs_b1'] <= .001
            result['windows'][mode][str(batch)] = row
    candidates = []
    for batch in [2,4,8,16]:
        rows = [result['train'][m][str(batch)] for m in ['S0','S1','S2']]
        if all(r['status']=='passed' and r['peak_reserved_mib'] <= .9*r['total_memory_mib'] for r in rows):
            candidates.append((sum(120/r['samples_per_second'] for r in rows), batch))
    assert candidates, 'No common stable training batch with >=10% reserve; stop rather than silently change architecture/precision'
    _, batch = min(candidates)
    window_candidates = []
    for window_batch in [1,4,8,16]:
        rows = [result['windows'][m][str(window_batch)] for m in ['S0','S1','S2']]
        if all(r['status']=='passed' and r['numerical_guard_passed'] and r['peak_reserved_mib'] <= .9*r['total_memory_mib'] for r in rows):
            window_candidates.append((sum(r['seconds_without_io_or_metrics'] for r in rows),window_batch))
    assert window_candidates, 'No common stable/equivalent window batch'
    fastest = min(window_candidates)[1]
    # Preserve exact historical selection protocol and avoid repeating an identical
    # 40-epoch baseline when training batch remains2. All window measurements retained.
    window_batch = 1 if batch == 2 else fastest
    result['selection'] = dict(train_batch=batch, window_batch=window_batch, fastest_measured_common_window_batch=fastest,
        train_rule='minimize sum of estimated120-sample epoch seconds across three modes; each reserved peak <=90% total memory',
        window_rule='If train batch2, preserve serial-window protocol for exact S0 reuse, saving a completed baseline; otherwise fastest common validated batch',
        candidate_train_epoch_seconds=candidates, candidate_combined_window_seconds=window_candidates,
        timed_partial_batches='train b16 benchmark skips final8 to time full16 updates; formal epochs preserve every final batch')
    write_json(BATCH/'throughput_summary.json', result)
    return result


def freeze(throughput):
    legacy_verify()
    verification = read(BATCH/'implementation_verification.json')
    assert verification['status'] == 'passed'
    assert read(BATCH/'recovery_verification.json')['status'] == 'passed'
    for path,digest in verification['source_sha256'].items():
        assert sha(ROOT/path) == digest, 'CPU preflight source mismatch '+path
    cuda_verification = read(BATCH/'cuda_verification.json')
    assert cuda_verification['status'] == 'passed' and cuda_verification['source_sha256'] == source_hashes(ROOT)
    assert read(ROOT/'outputs/analysis/b_zero_step_diagnostic_20261020/summary.json')['optimizer_steps'] == 0
    train_batch, window_batch = throughput['selection']['train_batch'], throughput['selection']['window_batch']
    configs = {m:dict(base_config(m, train_batch),window_batch_size=window_batch) for m in ['S0','S1','S2']}
    for mode, config in configs.items():
        path = ROOT/f'configs/pilot_static_semantic_{mode.lower()}.json'
        if path.exists():
            assert read(path) == config
        else:
            write_json(path, config)
    reuse = train_batch == 2 and window_batch == 1
    if reuse:
        audit = read(BATCH/'direct_reuse_audit.json')
        assert audit['status'] == 'passed'
        old = read(HISTORICAL_DIRECT/'config.json')
        keys = ['seed','manifest','crop_size','stride','batch_size','workers','epochs','warmup_epochs','lr','min_lr',
                'weight_decay','grad_clip','class_balance','boundary','boundary_metrics','boundary_tolerance','val_every','cpu_threads']
        assert all(old[k] == configs['S0'][k] for k in keys)
    runs = {m:str(ROOT/f'outputs/P1/static_semantic/seed_42/{BATCH.name}_{m.lower()}') for m in configs}
    if reuse:
        runs['S0'] = str(HISTORICAL_DIRECT)
    paths = [ROOT/'work/launch_static_semantic_20261020.py', ROOT/'work/static_semantic_worker.py', ROOT/'work/verify_static_cuda_20261020.py',
             ROOT/'work/profile_static_semantic_20261020.py', ROOT/'tests/test_static_recovery.py',
             ROOT/'work/verify_static_semantic_20261020.py', ROOT/'tests/test_static_semantic.py']
    paths += [BATCH/p for p in ['source_before.json','authorization.json','benchmark_protocol.json',
             'implementation_verification.json','cuda_verification.json','direct_reuse_audit.json','throughput_summary.json',
             'recovery_hardening_registration.json','recovery_verification.json','complexity.json']]
    paths += list((ROOT/'outputs/analysis/b_zero_step_diagnostic_20261020').glob('*.json'))
    paths += [ROOT/'work/diagnose_b_static_preflight_20261020.py']
    manifest = read_manifest(configs['S2']['manifest'])
    paths += [Path(configs['S2']['manifest'])]+[Path(x['snapshot_path']) for x in manifest['splits'].values()]
    plan = dict(batch_id=BATCH.name, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        authorization=read(BATCH/'authorization.json'), configs=configs,runs=runs,
        source_sha256=source_hashes(ROOT), workflow_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        source_change_set=sorted(set(source_hashes(ROOT))-set(read(BATCH/'source_before.json')['legacy_categories']['source_sha256'])),
        screening=SCREEN, budget=dict(epochs=40,train_seed=42,records_per_epoch=120,unique_train_rgb=119,
        sample_exposures_per_mode=4800,updates_per_mode=40*math.ceil(120/train_batch),train_batch=train_batch,
        new_training_count=2 if reuse else 3,reused_completed_direct=reuse),
        parameters=dict(S0=4929156,S1=5939156,S2=5939156),
        scope='Exactly MoS2 matched S0/S1/S2 scratch40 (S0 reused only if fully equivalent); train/val only; S2 samebest3408. Stop after reporting; no other material/test/200epochs/module queue.',
        precision='FP32 tensors; deterministic; matmul TF32 false, cuDNN TF32 true matching history',
        initialization='two independent CPU RNG domains both seed42; shared encoder/decoder/head identical; S1/S2 state identical',
        S0_equivalence='RGB/decoder architecture and initialized weights equal historical Direct. Do not claim new stronger RGB architecture versus historical Direct.',
        evaluation='full25val every10epochs, each native fixed mode readout; S2 five DDIM3407 selects best,3408 samebest only',
        dataset_caveat='original120 training records include119 unique RGB; m169/m56 duplicate image with929 differing annotation pixels; originals preserved')
    write_json(BATCH/'plan.json', plan)
    verify(plan)
    return plan


def audit_confusion(run, metrics):
    with np.load(Path(run)/'best_val_confusion.npz') as data:
        assert len(data['ids']) == 25 and np.array_equal(data['per_image'].sum(0),data['global_matrix'])
        values = summarize(data['global_matrix'])
        assert all(abs(values[k]-metrics[k]) < 1e-12 for k in values)


def report(plan, results):
    s0, s1, s2 = [results[m]['metrics'] for m in ['S0','S1','S2']]
    secondary = results['S2_secondary']['metrics']
    checks = dict(primary_vs_s0=s2['miou_4class']-s0['miou_4class'] >= .005,
                  primary_vs_s1=s2['miou_4class']-s1['miou_4class'] >= .003,
                  mean_thin_iou_gain=((s2['iou_monolayer']+s2['iou_fewlayer'])-(s0['iou_monolayer']+s0['iou_fewlayer']))/2 >= .005,
                  thin_and_boundary_guard=all(s2[k]-s0[k] >= -.005 for k in ['iou_monolayer','iou_fewlayer','boundary_f1_foreground']),
                  secondary_positive=all(secondary['miou_4class']-x['miou_4class'] > 0 for x in [s0,s1]))
    write_json(BATCH/'comparison_summary.json',dict(results=results,screening=SCREEN,screening_checks=checks,
        screening_passed=all(checks.values()),budget=plan['budget'],source_sha256=plan['source_sha256'],
        conclusion='Exploratory one-training-seed platform screen; retain all successes/failures and stop. No significance or established originality.'))
    lines = ['# 静态RGB语义条件平台筛选结果','', '| 模式 | best轮 | val mIoU % | 单层IoU % | 少层IoU % | 前景边界F1 % |',
             '|---|---:|---:|---:|---:|---:|']
    for mode in ['S0','S1','S2','S2_secondary']:
        row = results[mode]; m = row['metrics']
        lines.append(f"| {mode} | {row['epoch']} | {100*m['miou_4class']:.4f} | {100*m['iou_monolayer']:.4f} | {100*m['iou_fewlayer']:.4f} | {100*m['boundary_f1_foreground']:.4f} |")
    lines += ['', '探索性筛选：'+('通过' if all(checks.values()) else '未通过')+'。',
              'S2_secondary仅为同一best权重的3408推理复核，不是独立训练；未使用test。',
              '全部已登记训练完成，停止本批；不自动增加预算、模块或材料数据集。']
    (BATCH/'final_analysis.md').write_text('\n'.join(lines)+'\n')


def main():
    global STAGE
    lock = (ROOT/'outputs/.experiments.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        legacy_verify()
        if (BATCH/'plan.json').exists():
            plan = read(BATCH/'plan.json'); verify(plan)
        else:
            throughput = benchmarks()
            STAGE = 'freezing'; status('running')
            plan = freeze(throughput)
        results = {}
        for mode in ['S0','S1','S2']:
            verify(plan); STAGE = 'training_'+mode; status('running',budget=plan['budget'])
            run = Path(plan['runs'][mode])
            if mode == 'S0' and plan['budget']['reused_completed_direct']:
                assert read(run/'status.json')['status'] == 'completed'
                print('REUSE_COMPLETED_HISTORICAL_DIRECT',run,flush=True)
            else:
                current = {}
                try:
                    current = read(run/'status.json')
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                if current.get('status') != 'completed':
                    worker('train',mode,plan['budget']['train_batch'],BATCH/f'train_{mode}_result.json',BATCH/'plan.json')
            state = read(run/'status.json')
            assert state == dict(status='completed',last_completed_epoch=40,global_step=plan['budget']['updates_per_mode'])
            row = read(run/'best_val_metrics.json'); audit_confusion(run,row['metrics'])
            results[mode] = dict(**row,run=str(run),checkpoint_sha256=sha(run/'checkpoints/best.pt'))
            write_json(BATCH/'partial_results.json',results)
        STAGE = 'same_best_S2_3408'; status('running'); verify(plan)
        checkpoint = Path(plan['runs']['S2'])/'checkpoints/best.pt'; digest = sha(checkpoint)
        path = BATCH/'secondary_S2_3408.json'
        previous = None
        if path.exists():
            try:
                previous = read(path)
            except json.JSONDecodeError:
                pass
            if not (isinstance(previous,dict) and previous.get('status') == 'completed'
                    and previous.get('epoch') == results['S2']['epoch'] and previous.get('sampling_seed') == 3408
                    and 'metrics' in previous):
                stamp = str(time.time_ns())
                for artifact in [path,path.with_suffix('.npz'),path.with_name(path.stem+'_timing.json')]:
                    if artifact.exists():
                        artifact.rename(artifact.with_name(artifact.stem+'_failed_attempt_'+stamp+artifact.suffix))
                previous = None
        results['S2_secondary'] = previous or worker('secondary','S2',plan['budget']['train_batch'],path,BATCH/'plan.json')
        assert results['S2_secondary']['epoch'] == results['S2']['epoch'] and sha(checkpoint) == digest
        STAGE = 'reporting'; status('running'); report(plan,results); verify(plan)
        STAGE = 'finished'; status('completed',screening_passed=read(BATCH/'comparison_summary.json')['screening_passed'])
    except BaseException as exc:
        status('failed',error=str(exc)); traceback.print_exc(); raise


if __name__ == '__main__':
    main()

"""Exactly two seed42 frozen-reference adapters; validate, report, and stop."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import csv
import fcntl
import hashlib
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.runtime import source_hashes, write_json

BATCH = ROOT / 'outputs/launches/p1_reference_adapter_20261018'
OLD = ROOT / 'outputs/launches/p1_reference_fusion_20261018'
RUNS = {v: ROOT / f'outputs/P1/diffusion/seed_42/p1_reference_adapter_20261018_{v}' for v in ['a', 'b']}
SAMPLING = dict(sampling_steps=20, samples_per_image=1, sampling_seed=3408)
STAGE = 'preflight'
SCREEN = dict(primary_B_minus_F0_minimum_miou_gain=.005, primary_B_minus_A_minimum_miou_gain=.003,
              both_secondary_miou_gains_strictly_positive=True, monolayer_both_seeds_improve=True,
              maximum_few_or_boundary_decline_vs_F0=.005, rejected_predictions_exactly_preserved=True,
              interpretation='Exploratory one-training-seed pilot, not significance or established novelty; original F1 failure remains failed.')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(value, **extra):
    write_json(BATCH / 'status.json', dict(status=value, stage=STAGE, pid=os.getpid(),
               updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra))


def verify(plan):
    assert source_hashes(ROOT) == plan['source_sha256'], 'Frozen new source/config changed'
    for category in ['protected_artifacts_sha256', 'workflow_sha256', 'preflight_artifacts_sha256']:
        for path, digest in plan[category].items():
            assert sha(ROOT / path) == digest, category + ': ' + path
    assert read(BATCH / 'validation_protocol3408.json') == SAMPLING


def prepare():
    if (BATCH / 'plan.json').exists():
        plan = read(BATCH / 'plan.json'); verify(plan); return plan
    before = read(BATCH / 'source_before.json'); frozen = source_hashes(ROOT)
    changed = sorted(k for k in set(frozen) | set(before['source_before_sha256'])
                     if frozen.get(k) != before['source_before_sha256'].get(k))
    assert changed == sorted(['src/conditional_diffusion/model.py', 'src/conditional_diffusion/trainer.py',
                             'src/conditional_diffusion/reference_data.py', 'src/conditional_diffusion/reference_adapter.py',
                             'configs/pilot_reference_adapter_a.json', 'configs/pilot_reference_adapter_b.json']), changed
    records = ['legacy_verification.json', 'resource_verification.json', 'resume_verification.json', 'report_verification.json',
               'fallback_verification.json', 'preflight_completed.json']
    assert all(read(BATCH / p)['status'] == 'passed' for p in records)
    assert len(read(BATCH / 'fallback_verification.json')['results']) == 20
    tests = (BATCH / 'unit_tests_attempt1.log').read_text()
    assert 'Ran 46 tests' in tests and tests.rstrip().endswith('OK')
    configs = {v: read(ROOT / f'configs/pilot_reference_adapter_{v}.json') for v in RUNS}
    diff = {k: [configs['a'].get(k), configs['b'].get(k)] for k in set(configs['a']) | set(configs['b'])
            if configs['a'].get(k) != configs['b'].get(k)}
    assert set(diff) == {'adapter_reference_enabled', 'experiment_variant'}
    for c in configs.values():
        assert c['seed'] == 42 and c['epochs'] == 40 and c['batch_size'] == 2
        assert c['base_channels'] == c['adapter_width'] == 32 and c['crop_size'] == 512 and c['stride'] == 384
        assert c['sampling_seed'] == 3407 and c['sampling_steps'] == 20 and c['samples_per_image'] == 1
        assert sha(c['adapter_base_checkpoint']) == c['adapter_base_sha256']
        assert sha(c['reference_cache']) == c['reference_cache_sha256']
    workflow_paths = ['work/launch_p1_reference_adapter.py', 'work/verify_reference_adapter.py', 'work/queue_reference_adapter.py']
    workflow_paths += [str(p.relative_to(ROOT)) for p in (ROOT / 'tests').glob('test_*.py')]
    workflow_paths += [str(p.relative_to(ROOT)) for p in Path(before['backup']).rglob('*') if p.is_file()]
    workflows = dict(before['previous_workflow_sha256'])
    workflows.update({p: sha(ROOT / p) for p in workflow_paths})
    for p, digest in before['previous_workflow_sha256'].items():
        assert sha(ROOT / p) == digest
    protected = dict(before['protected_artifacts_sha256'])
    protected.update(before['previous_reference_sha256'])
    reference_snapshot = read(OLD / 'comparison_summary.json')['results']['f0']
    plan = dict(batch_id=BATCH.name, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                authorization='User explicitly requested start training after canceling multi-training-seed work and reviewing concrete optimizations. First-stage A/B only.',
                source_sha256=frozen, source_change_set=changed,
                protected_artifacts_sha256=protected, workflow_sha256=workflows,
                preflight_artifacts_sha256={str((BATCH / p).relative_to(ROOT)): sha(BATCH / p)
                                           for p in records + ['unit_tests_attempt1.log', 'source_before.json', 'queued_request.json']},
                configs=configs, config_difference=diff, runs={v: str(p) for v, p in RUNS.items()},
                parameters=5548392, trainable_parameters=26052, base_parameters=5522340,
                scope='Exactly A then B, each40 adapter epochs/2400 planned batches, seed42, frozen F0 seed42 best40; full25val20DDIM3407 every10 selects best, samebest3408. Report and stop. No dynamic gate, crop/loss changes, unfreeze C, other training seeds, formal or test queue.',
                feature_definition='Original frozen RGB-only reference cache; unweighted clipped C plus q. Base always receives RGB+zeroaux. A zeroes C only within adapter; q/data/params/init unchanged. q is applied once at residual output, hard bypass when rejected.',
                all_rejected_batch_policy='Log loss, skip backward and optimizer.step including AdamW moments/decay; planned global_step advances, epoch scheduler unchanged. Log active exposures and actual optimizer/nonzero-gradient updates.',
                base_input_and_freeze='Base eval, requires_grad false, no_grad forward. Base parameters verified after synthetic updates/integration and again in best/last before secondary evaluation. Rejected paths compare numerically at every DDIM step against independent F0 for all10 val images at both seeds in preflight.',
                coverage=dict(train_accepted=71, train_total=120, val_accepted=15, val_total=25),
                total_training_budget='40 base epochs plus 40 adapter epochs, not a from-scratch40 comparison with historical D/I/HR.',
                F0=reference_snapshot, screening=SCREEN,
                references='Generic frozen/zero-initialized conditioning has prior art. Posthoc old subgroup findings are motivation only; new full-val gate predeclared. No significance or established innovation claim.')
    write_json(BATCH / 'validation_protocol3408.json', SAMPLING)
    write_json(BATCH / 'plan.json', plan); verify(plan)
    return plan


def epoch_values(run, tag):
    with (run / 'metrics_epochs.csv').open() as f:
        return {int(r['epoch']): float(r['value']) for r in csv.DictReader(f) if r['tag'] == tag}


def audit_base(run, plan):
    base = torch.load(plan['configs']['a']['adapter_base_checkpoint'], map_location='cpu', weights_only=False)['model']
    for name in ['best', 'last']:
        checkpoint = torch.load(run / f'checkpoints/{name}.pt', map_location='cpu', weights_only=False)
        assert all(torch.equal(value, checkpoint['model']['base.' + key]) for key, value in base.items()), 'Frozen F0 weights changed'
    return True


def evaluate_secondary(v, plan):
    verify(plan); run = RUNS[v]; checkpoint = run / 'checkpoints/best.pt'; digest = sha(checkpoint)
    record_path = BATCH / (v + '_seed3408.json')
    if record_path.exists():
        record = read(record_path)
        assert record['checkpoint_sha256'] == digest and record['evaluation_source_sha256'] == plan['source_sha256']
        return record
    before = set((run / 'evaluations').glob('val_*'))
    with (BATCH / (v + '_seed3408.log')).open('a') as log:
        subprocess.run([sys.executable, '-u', 'scripts/evaluate.py', '--checkpoint', str(checkpoint), '--split', 'val',
                        '--full-val', '--protocol', str(BATCH / 'validation_protocol3408.json')],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    added = set((run / 'evaluations').glob('val_*')) - before
    assert len(added) == 1
    output = added.pop(); metrics = read(output / 'metrics.json'); ec = read(output / 'evaluation_config.json')
    assert metrics['image_count'] == 25 and metrics['nfe_total'] == 17500 and ec['split'] == 'val'
    assert ec['source_sha256'] == plan['source_sha256'] and all(ec['config'][k] == x for k, x in SAMPLING.items())
    assert ec['checkpoint_epoch'] == read(run / 'best_val_metrics.json')['epoch'] and sha(checkpoint) == digest
    record = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, checkpoint_epoch=ec['checkpoint_epoch'],
                  evaluation_directory=str(output), metrics=metrics, evaluation_source_sha256=plan['source_sha256'])
    verify(plan); write_json(record_path, record); return record


def groups(v, seed, result, plan):
    from conditional_diffusion.metrics_impl import summarize
    run = RUNS[v]
    candidate_path = run / 'best_val_confusion.npz' if seed == '3407' else Path(result['evaluation_directory']) / 'confusion_matrices.npz'
    base_path = Path(plan['configs']['a']['adapter_base_checkpoint']).parent.parent / 'best_val_confusion.npz' if seed == '3407' else Path(plan['F0']['3408']['evaluation_directory']) / 'confusion_matrices.npz'
    with np.load(candidate_path) as data:
        ids, matrices = data['ids'].copy(), data['per_image'].copy()
        assert len(ids) == 25 and np.array_equal(matrices.sum(0), data['global_matrix'])
        recomputed = summarize(data['global_matrix'])
        assert all(abs(recomputed[k] - result['metrics'][k]) < 1e-12 for k in recomputed)
    with np.load(base_path) as data:
        previous = {str(i): m.copy() for i, m in zip(data['ids'], data['per_image'])}
    cache = read(plan['configs']['a']['reference_cache'])['images']
    group_results = {}
    for accepted, name in [(True, 'accepted'), (False, 'rejected')]:
        indices = [j for j, ident in enumerate(ids) if cache[str(ident)]['accepted'] == accepted]
        old = np.sum([previous[str(ids[j])] for j in indices], axis=0)
        new = matrices[indices].sum(0)
        if not accepted:
            assert all(np.array_equal(matrices[j], previous[str(ids[j])]) for j in indices), 'Rejected fallback matrix differs'
        group_results[name] = dict(image_count=len(indices), F0=summarize(old), candidate=summarize(new))
    return group_results


def finish(plan, results):
    keys = ['miou_4class', 'iou_monolayer', 'iou_fewlayer', 'boundary_f1_foreground']
    deltas = {pair: {seed: {k: results['b'][seed]['metrics'][k] - baseline[seed]['metrics'][k] for k in keys}
                      for seed in ['3407', '3408']}
              for pair, baseline in [('B_minus_F0', plan['F0']), ('B_minus_A', results['a'])]}
    checks = dict(primary_vs_F0=deltas['B_minus_F0']['3407']['miou_4class'] >= .005,
                  primary_vs_A=deltas['B_minus_A']['3407']['miou_4class'] >= .003,
                  secondary_positive=all(deltas[p]['3408']['miou_4class'] > 0 for p in deltas),
                  mono_improves=all(d['iou_monolayer'] > 0 for d in deltas['B_minus_F0'].values()),
                  few_and_boundary_guard=all(d[k] >= -.005 for d in deltas['B_minus_F0'].values() for k in ['iou_fewlayer', 'boundary_f1_foreground']))
    for tag in ['diffusion_train/timestep_mean', 'diffusion_train/high_noise_fraction', 'diffusion_train/recycle_fraction',
                'diffusion_train/recycled_high_noise_fraction', 'reference_train/active_fraction', 'adapter_train/optimizer_updates',
                'adapter_train/all_rejected_batches', 'adapter_train/accepted_sample_exposures']:
        assert epoch_values(RUNS['a'], tag) == epoch_values(RUNS['b'], tag), tag
    subgroup = {v: {seed: groups(v, seed, results[v][seed], plan) for seed in ['3407', '3408']} for v in RUNS}
    checks['rejected_preserved'] = True
    write_json(BATCH / 'reference_group_diagnosis.json', subgroup)
    write_json(BATCH / 'comparison_summary.json', dict(results=results, F0=plan['F0'], deltas=deltas,
               screening=SCREEN, screening_checks=checks, screening_passed=all(checks.values()),
               base_unchanged=True, training_sequence_counters_equal=True, source_sha256=plan['source_sha256'], scope=plan['scope']))
    lines = ['# Frozen F0 reference residual A/B', '',
             '| Model | Best adapter epoch | mIoU3407 % | mIoU3408 % | Mono3407 % | Few3407 % | Boundary3407 % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for v, row in [('F0', plan['F0']), ('A', results['a']), ('B', results['b'])]:
        primary = row['3407']; m = primary['metrics']; secondary = row['3408']['metrics']
        lines.append(f"| {v} | {primary['epoch']} | {100*m['miou_4class']:.2f} | {100*secondary['miou_4class']:.2f} | {100*m['iou_monolayer']:.2f} | {100*m['iou_fewlayer']:.2f} | {100*m['boundary_f1_foreground']:.2f} |")
    lines += ['', 'Exploratory screen: ' + ('passed' if all(checks.values()) else 'failed') + '.',
              'Single trainingseed42;3408 evaluates samebest. Base40+adapter40 cumulative budget. No significance or established novelty claim.',
              'Rejected group is held at F0 predictions; subgroup improvements do not replace the full-val screen.',
              'All planned work completed. Stop; no gate/crop/loss/unfreeze/multi-training-seed/formal/test queue. Retain original F1 failure.']
    (BATCH / 'final_analysis.md').write_text('\n'.join(lines) + '\n')


def main():
    global STAGE
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    lock = (ROOT / 'outputs/.experiments.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        plan = prepare()
        if args.prepare_only:
            print('ADAPTER_PLAN_FROZEN', BATCH / 'plan.json', flush=True); return
        if (BATCH / 'status.json').exists() and read(BATCH / 'status.json')['status'] == 'completed':
            return
        results = {}
        for v, run in RUNS.items():
            verify(plan); STAGE = 'training_' + v; status('running')
            if run.exists():
                assert read(run / 'config.json') == plan['configs'][v]
                assert read(run / 'run_metadata.json')['source_sha256'] == plan['source_sha256']
                if read(run / 'status.json')['status'] != 'completed':
                    subprocess.run([sys.executable, '-u', 'scripts/train.py', '--resume', str(run / 'checkpoints/last.pt')], cwd=ROOT, check=True)
            else:
                subprocess.run([sys.executable, '-u', 'scripts/train.py', '--config', str(ROOT / f'configs/pilot_reference_adapter_{v}.json'),
                                '--run-id', run.name], cwd=ROOT, check=True)
            assert read(run / 'status.json') == dict(status='completed', last_completed_epoch=40, global_step=2400)
            for tag, value in [('diffusion_train/high_noise_fraction', .5), ('diffusion_train/recycle_fraction', .5),
                               ('reference_train/active_fraction', 71/120)]:
                values = epoch_values(run, tag); assert len(values) == 40 and all(x == value for x in values.values())
            assert sum(epoch_values(run, 'adapter_train/accepted_sample_exposures').values()) == 2840
            updates = sum(epoch_values(run, 'adapter_train/optimizer_updates').values())
            skipped = sum(epoch_values(run, 'adapter_train/all_rejected_batches').values())
            assert updates + skipped == 2400 and skipped > 0
            audit_base(run, plan); verify(plan); STAGE = v + '_seed3408'; status('running')
            results[v] = {'3407': read(run / 'best_val_metrics.json'), '3408': evaluate_secondary(v, plan)}
            write_json(BATCH / 'partial_results.json', results)
        STAGE = 'reporting'; status('running'); finish(plan, results); verify(plan)
        STAGE = 'finished'; status('completed', screening_passed=read(BATCH / 'comparison_summary.json')['screening_passed'])
    except BaseException as exc:
        status('failed', error=str(exc)); traceback.print_exc(); raise


if __name__ == '__main__':
    main()

"""Bounded F0/F1 pilot after RGB-reference audit; stop after validation comparison."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.runtime import source_hashes, write_json

BATCH = ROOT / 'outputs/launches/p1_reference_fusion_20261018'
STAGE = 'preflight'
RUNS = {v: ROOT / f'outputs/P1/diffusion/seed_42/p1_reference_fusion_20261018_{v}' for v in ['f0', 'f1']}
SAMPLING = dict(sampling_steps=20, samples_per_image=1, sampling_seed=3408)
SCREEN = dict(minimum_primary_miou_gain=0.005, secondary_miou_gain_strictly_positive=True,
              minority_mean_iou_no_decline=True, maximum_individual_minority_iou_decline=0.01,
              maximum_miou_gap_below_historical_HR=0.01,
              interpretation='Exploratory predeclared pilot screen, not significance or novelty. Complete F0/F1 regardless of relative performance; report and stop. No F2/F3/D-C auto queue.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def status(value, **extra):
    write_json(BATCH / 'status.json', dict(status=value, stage=STAGE, pid=os.getpid(),
               updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra))


def verify(plan):
    assert source_hashes(ROOT) == plan['source_sha256'], 'Frozen new source/config changed'
    for category in ['protected_artifacts_sha256', 'workflow_sha256', 'reference_artifacts_sha256']:
        for path, digest in plan[category].items():
            assert sha(ROOT / path) == digest, category + ': ' + path
    assert read(BATCH / 'validation_protocol3408.json') == SAMPLING


def prepare():
    path = BATCH / 'plan.json'
    if path.exists():
        plan = read(path);verify(plan);return plan
    before = read(BATCH / 'source_before.json')
    frozen = source_hashes(ROOT)
    changed = sorted(k for k in set(before['source_before_sha256']) | set(frozen)
                     if before['source_before_sha256'].get(k) != frozen.get(k))
    expected = ['configs/pilot_reference_f0.json', 'configs/pilot_reference_f1.json', 'scripts/evaluate.py',
                'src/conditional_diffusion/condition_model.py', 'src/conditional_diffusion/reference_data.py',
                'src/conditional_diffusion/model.py', 'src/conditional_diffusion/inference.py', 'src/conditional_diffusion/trainer.py']
    assert changed == sorted(expected), changed
    assert read(BATCH / 'reference_audit.json')['passed']
    for name in ['legacy_verification.json', 'resource_verification.json', 'resume_verification.json']:
        assert read(BATCH / name)['status'] == 'passed'
    unit_log = (BATCH / 'unit_tests_attempt1.log').read_text()
    assert 'Ran 40 tests' in unit_log and unit_log.rstrip().endswith('OK')
    configs = {v: read(ROOT / f'configs/pilot_reference_{v}.json') for v in ['f0', 'f1']}
    difference = {k: [configs['f0'].get(k), configs['f1'].get(k)] for k in set(configs['f0']) | set(configs['f1'])
                  if configs['f0'].get(k) != configs['f1'].get(k)}
    assert set(difference) == {'reference_enabled', 'experiment_variant'}
    cache_path = Path(configs['f0']['reference_cache']);cache = read(cache_path)
    assert sha(cache_path) == configs['f0']['reference_cache_sha256']
    manifest = read(ROOT / configs['f0']['manifest'])
    expected_ids = set()
    for split in ['train', 'val']:
        expected_ids.update(Path(manifest['splits'][split]['snapshot_path']).read_text().split())
    assert set(cache['images']) == expected_ids and len(expected_ids) == 145
    for ident, entry in cache['images'].items():
        rgb = Path(manifest['dataset_root']) / manifest['image_pattern'].format(id=ident)
        assert sha(rgb) == entry['rgb_sha256']
    for c in configs.values():
        assert c['seed'] == 42 and c['epochs'] == 40 and c['batch_size'] == 2
        assert c['crop_size'] == 512 and c['stride'] == 384 and c['sampling_seed'] == 3407
        assert c['sampling_steps'] == 20 and c['samples_per_image'] == 1
    workflows = ['work/launch_p1_reference_fusion.py', 'work/verify_reference_fusion.py', 'work/audit_rgb_reference.py']
    workflows += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'tests').glob('test_*.py'))]
    workflows += [str(p.relative_to(ROOT)) for p in sorted(Path(before['backup']).rglob('*'))
                  if p.is_file() and p.suffix in {'.py', '.sh', '.json'}]
    reference_files = ['reference_audit_plan.json', 'reference_audit.json', 'reference_cache.json',
                       'legacy_verification.json', 'resource_verification.json', 'resume_verification.json', 'unit_tests_attempt1.log']
    plan = dict(batch_id=BATCH.name, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                authorization='User requested implementation and execution of the proposed algorithm direction; bounded first-stage F0/F1 pilot.',
                source_sha256=frozen, source_change_set=changed,
                protected_artifacts_sha256=before['protected_artifacts_sha256'],
                workflow_sha256={p: sha(ROOT / p) for p in workflows},
                reference_artifacts_sha256={str((BATCH / p).relative_to(ROOT)): sha(BATCH / p) for p in reference_files},
                configs=configs, config_difference=difference, runs={k: str(v) for k, v in RUNS.items()},
                parameters=5522340, historical_HR_parameters=5132676,
                scope='Exactly two from-scratch40-epoch P1 runs: F0 common split-image/state encoder with ordinary mean fusion and zero reference features; F1 identical architecture with RGB-only q*C/q features. Each best then full25-val3408. Report/screen and stop. No test, other seed, F2/F3/D-C, old all flow or formal matrix.',
                feature_definition='Full-image64x64RGB quadratic Tukey IRLS12; confidence/support thresholds frozen in train-only audit. Native patch coordinates; C clipped[-1,1], weighted by accepted RGB fit confidence q; rejected references produce four zeros. Same crop/augmentation/data order; no label inputs to feature construction.',
                coverage={'train': dict(accepted=71,total=120), 'val': dict(accepted=15,total=25)},
                selection='Full25val20DDIM/one sample/seed3407 every10epochs;3408 evaluates samebest without reselection.',
                reporting='Per-class IoU/F1, foreground boundaryF1, confusion matrices, real source/config/cache and trained-source provenance, train/validation times, params, high-noise/recycling/reference usage; single training seed, no significance/novelty claim.',
                screening=SCREEN,
                comparability='F0/F1 have exact matching initial parameters and initialization RNG. Dataset transformations and training time/noise RNG should match. Both differ architecturally and by initialization RNG from historicalHR; do not attribute F0-vsHR differences solely to one feature.',
                historical_HR=dict(run=str(ROOT / 'outputs/P1/diffusion/seed_42/p1_recycling_20261017'),
                    primary=read(ROOT / 'outputs/P1/diffusion/seed_42/p1_recycling_20261017/best_val_metrics.json'),
                    secondary=read(ROOT / 'outputs/launches/p1_recycling_20261017/candidate_seed3408.json'),
                    trained_source_sha256=read(ROOT / 'outputs/P1/diffusion/seed_42/p1_recycling_20261017/run_metadata.json')['source_sha256']))
    write_json(BATCH / 'validation_protocol3408.json', SAMPLING)
    write_json(path, plan);verify(plan)
    return plan


def evaluate_secondary(variant, plan):
    verify(plan);run = RUNS[variant];checkpoint = run / 'checkpoints/best.pt';digest = sha(checkpoint)
    cache = BATCH / (variant + '_seed3408.json')
    if cache.exists():
        record = read(cache)
        assert record['checkpoint_sha256'] == digest and record['evaluation_source_sha256'] == plan['source_sha256']
        return record
    before = set((run / 'evaluations').glob('val_*'))
    with (BATCH / (variant + '_seed3408.log')).open('a') as log:
        subprocess.run([sys.executable, '-u', 'scripts/evaluate.py', '--checkpoint', str(checkpoint),
                        '--split', 'val', '--full-val', '--protocol', str(BATCH / 'validation_protocol3408.json')],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    added = set((run / 'evaluations').glob('val_*')) - before
    assert len(added) == 1
    out = added.pop();metrics, ec = read(out / 'metrics.json'), read(out / 'evaluation_config.json')
    assert metrics['image_count'] == 25 and metrics['nfe_total'] == 17500 and ec['split'] == 'val'
    assert ec['source_sha256'] == plan['source_sha256'] and all(ec['config'][k] == v for k, v in SAMPLING.items())
    assert ec['config']['crop_size'] == 512 and ec['config']['stride'] == 384
    assert ec['checkpoint_epoch'] == read(run / 'best_val_metrics.json')['epoch'] and sha(checkpoint) == digest
    record = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, checkpoint_epoch=ec['checkpoint_epoch'],
                  evaluation_directory=str(out), metrics=metrics, evaluation_source_sha256=plan['source_sha256'])
    verify(plan);write_json(cache, record);return record


def epoch_values(run, tag):
    with (run / 'metrics_epochs.csv').open() as f:
        return {int(r['epoch']): float(r['value']) for r in csv.DictReader(f) if r['tag'] == tag}


def finish(plan, results):
    pairs = {seed: (results['f0'][seed]['metrics'], results['f1'][seed]['metrics']) for seed in ['3407', '3408']}
    deltas = {s: {k: b[k] - a[k] for k in ['miou_4class', 'iou_monolayer', 'iou_fewlayer', 'boundary_f1_foreground']}
              for s, (a, b) in pairs.items()}
    historical = {'3407': plan['historical_HR']['primary']['metrics'], '3408': plan['historical_HR']['secondary']['metrics']}
    checks = dict(primary_gain=deltas['3407']['miou_4class'] >= SCREEN['minimum_primary_miou_gain'],
                  secondary_positive=deltas['3408']['miou_4class'] > 0,
                  minority_average_no_decline=all((d['iou_monolayer'] + d['iou_fewlayer']) / 2 >= 0 for d in deltas.values()),
                  individual_minority_guard=all(d[k] >= -SCREEN['maximum_individual_minority_iou_decline'] for d in deltas.values() for k in ['iou_monolayer', 'iou_fewlayer']),
                  near_historical_HR=all(results['f1'][s]['metrics']['miou_4class'] >= historical[s]['miou_4class'] - SCREEN['maximum_miou_gap_below_historical_HR'] for s in pairs))
    timing_sequence_equal = epoch_values(RUNS['f0'], 'diffusion_train/timestep_mean') == epoch_values(RUNS['f1'], 'diffusion_train/timestep_mean')
    assert timing_sequence_equal
    write_json(BATCH / 'comparison_summary.json', dict(results=results, deltas_f1_minus_f0=deltas,
               screening_checks=checks, screening_passed=all(checks.values()), screening=SCREEN,
               timestep_sequence_summary_equal=timing_sequence_equal, historical_HR=plan['historical_HR'],
               source_sha256=plan['source_sha256'], scope=plan['scope']))
    lines = ['# F0/F1 reference feature pilot', '',
             '| Model | Best epoch | mIoU3407 % | mIoU3408 % | Mono IoU3407 % | Few IoU3407 % | Foreground boundaryF1 3407 % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for v in ['f0', 'f1']:
        a, b = results[v]['3407'], results[v]['3408'];m = a['metrics']
        lines.append(f"| {v} | {a['epoch']} | {m['miou_4class']*100:.2f} | {b['metrics']['miou_4class']*100:.2f} | {m['iou_monolayer']*100:.2f} | {m['iou_fewlayer']*100:.2f} | {m['boundary_f1_foreground']*100:.2f} |")
    lines += ['', 'Predeclared exploratory screen: ' + ('passed' if all(checks.values()) else 'not passed') + '.',
              'No statistical significance or established novelty claim; one training seed42. Complete source/cache/protected-weight hashes verified.3408 never reselects best.', '',
              'This batch stops here. F2/F3/Direct-reference, other training seeds, formal runs and any test evaluation were not queued. Preserve negative results and the gate.']
    (BATCH / 'final_analysis.md').write_text('\n'.join(lines) + '\n')


def main():
    global STAGE
    parser = argparse.ArgumentParser();parser.add_argument('--prepare-only', action='store_true');args = parser.parse_args()
    lock = (ROOT / 'outputs/.experiments.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        plan = prepare()
        if args.prepare_only:
            print('REFERENCE_PLAN_FROZEN', BATCH / 'plan.json', flush=True);return
        if (BATCH / 'status.json').exists() and read(BATCH / 'status.json')['status'] == 'completed':
            return
        results = {}
        for variant in ['f0', 'f1']:
            verify(plan);STAGE = 'training_' + variant;status('running')
            run = RUNS[variant]
            if run.exists():
                assert read(run / 'config.json') == plan['configs'][variant]
                assert read(run / 'run_metadata.json')['source_sha256'] == plan['source_sha256']
                if read(run / 'status.json')['status'] != 'completed':
                    subprocess.run([sys.executable, '-u', 'scripts/train.py', '--resume', str(run / 'checkpoints/last.pt')], cwd=ROOT, check=True)
            else:
                subprocess.run([sys.executable, '-u', 'scripts/train.py', '--config', str(ROOT / f'configs/pilot_reference_{variant}.json'),
                                '--run-id', run.name], cwd=ROOT, check=True)
            state = read(run / 'status.json')
            assert state == dict(status='completed', last_completed_epoch=40, global_step=2400)
            for tag, value in [('diffusion_train/high_noise_fraction', .5), ('diffusion_train/recycle_fraction', .5),
                               ('reference_train/active_fraction', 0.0 if variant == 'f0' else 71 / 120)]:
                values = epoch_values(run, tag);assert len(values) == 40 and all(x == value for x in values.values())
            assert sum(epoch_values(run, 'diffusion_train/teacher_sample_forwards').values()) == 2400
            verify(plan);STAGE = variant + '_seed3408';status('running')
            results[variant] = {'3407': read(run / 'best_val_metrics.json'), '3408': evaluate_secondary(variant, plan)}
            write_json(BATCH / 'partial_results.json', results)
        STAGE = 'reporting';status('running');verify(plan);finish(plan, results)
        STAGE = 'finished';status('completed', screening_passed=read(BATCH / 'comparison_summary.json')['screening_passed'])
    except BaseException as exc:
        status('failed', error=str(exc));traceback.print_exc();raise


if __name__ == '__main__':
    main()

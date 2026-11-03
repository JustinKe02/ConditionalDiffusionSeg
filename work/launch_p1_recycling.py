"""Bounded queue: four baseline step evaluations, one recycling P1, one extra seed."""
import os
import sys
import json
import csv
import hashlib
import subprocess
import fcntl
import traceback
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.runtime import write_json, source_hashes

BATCH_ID = 'p1_recycling_20261017'
BATCH = ROOT / 'outputs/launches' / BATCH_ID
BASE = ROOT / 'outputs/P1/diffusion/seed_42/p1_high_noise_20261017'
ORIGINAL = ROOT / 'outputs/P1/diffusion/seed_42/p1_cosine_20261017'
RUN = ROOT / 'outputs/P1/diffusion/seed_42' / BATCH_ID
STAGE = 'preflight'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(value, **extra):
    write_json(BATCH / 'status.json', dict(status=value, stage=STAGE, pid=os.getpid(),
                                         updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra))


def command(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def protected_hashes():
    return {str(run / 'checkpoints/best.pt'): sha(run / 'checkpoints/best.pt')
            for run in [BASE, ORIGINAL]}


def verify(plan):
    if source_hashes(ROOT) != plan['source_sha256']:
        raise RuntimeError('Frozen training/evaluation source or config changed')
    if protected_hashes() != plan['protected_checkpoint_sha256']:
        raise RuntimeError('A protected baseline checkpoint changed')
    if any(sha(ROOT / path) != digest for path, digest in plan['workflow_sha256'].items()):
        raise RuntimeError('Frozen workflow or test source changed')


def evaluate(run, label, steps, seed, plan):
    verify(plan)
    checkpoint = run / 'checkpoints/best.pt'
    digest = sha(checkpoint)
    protocol = dict(crop_size=512, stride=384, sampling_steps=steps,
                    samples_per_image=1, sampling_seed=seed)
    cache = BATCH / (label + '.json')
    if cache.exists():
        saved = json.loads(cache.read_text())
        if (saved['checkpoint_sha256'] != digest or saved['protocol'] != protocol
                or saved['evaluation_source_sha256'] != plan['source_sha256']):
            raise RuntimeError('Evaluation cache differs from frozen checkpoint/protocol/source')
        return saved
    before = set((run / 'evaluations').glob('val_*'))
    with (BATCH / (label + '.log')).open('a') as log:
        subprocess.run([sys.executable, '-u', 'scripts/evaluate.py', '--checkpoint', str(checkpoint),
                        '--split', 'val', '--full-val', '--window-size', '512', '--stride', '384',
                        '--steps', str(steps), '--samples', '1', '--sampling-seed', str(seed)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    added = set((run / 'evaluations').glob('val_*')) - before
    if len(added) != 1:
        raise RuntimeError('Cannot identify unique evaluation directory')
    out = added.pop()
    metrics = json.loads((out / 'metrics.json').read_text())
    ec = json.loads((out / 'evaluation_config.json').read_text())
    assert metrics['image_count'] == 25 and metrics['nfe_total'] == 875 * steps
    assert ec['split'] == 'val' and all(ec['config'][k] == v for k, v in protocol.items())
    assert ec['source_sha256'] == plan['source_sha256']
    assert sha(checkpoint) == digest
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, checkpoint_epoch=ec['checkpoint_epoch'],
                  protocol=protocol, evaluation_directory=str(out), metrics=metrics,
                  evaluation_source_sha256=plan['source_sha256'],
                  trained_source_sha256=json.loads((run / 'run_metadata.json').read_text())['source_sha256'])
    write_json(cache, report)
    print('EVALUATION_COMPLETED', label, json.dumps(metrics), flush=True)
    verify(plan)
    return report


def main():
    global STAGE
    BATCH.mkdir(parents=True, exist_ok=True)
    lock = open(ROOT / 'outputs/.experiments.lock', 'a')
    # Acquire before touching batch state; a duplicate launcher cannot falsify live status.
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        frozen = source_hashes(ROOT)
        before = json.loads((BATCH / 'source_before.json').read_text())
        changed = {k for k in set(before) | set(frozen) if before.get(k) != frozen.get(k)}
        assert changed == {'src/conditional_diffusion/trainer.py', 'src/conditional_diffusion/recycling.py',
                           'configs/pilot_recycling.json'}, changed
        previous_plan = json.loads((ROOT / 'outputs/launches/p1_high_noise_20261017/plan.json').read_text())
        assert before == previous_plan['source_sha256']
        assert sha(ORIGINAL / 'checkpoints/best.pt') == previous_plan['baseline_checkpoint_sha256']
        previous_result = json.loads((ROOT / 'outputs/launches/p1_high_noise_20261017/candidate_seed3408.json').read_text())
        assert sha(BASE / 'checkpoints/best.pt') == previous_result['checkpoint_sha256']
        baseline_config = json.loads((BASE / 'config.json').read_text())
        config = json.loads((ROOT / 'configs/pilot_recycling.json').read_text())
        diff = {k: [baseline_config.get(k), config.get(k)] for k in set(baseline_config) | set(config)
                if baseline_config.get(k) != config.get(k)}
        assert set(diff) == {'training_state', 'recycle_fraction', 'experiment_variant'}
        assert config['seed'] == 42 and config['epochs'] == 40 and config['batch_size'] == 2
        for run in [BASE, ORIGINAL]:
            assert json.loads((run / 'status.json').read_text())['status'] == 'completed'
        workflows = ['work/launch_p1_recycling.py', 'work/verify_p1_recycling_resume.py']
        workflows += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'tests').glob('test_*.py'))]
        plan = dict(batch_id=BATCH_ID, source_sha256=frozen, source_change_set=sorted(changed),
                    workflow_sha256={p: sha(ROOT / p) for p in workflows}, config=config, config_diff=diff,
                    protected_checkpoint_sha256=protected_hashes(), baseline_run=str(BASE),
                    baseline_trained_source_sha256=json.loads((BASE / 'run_metadata.json').read_text())['source_sha256'],
                    initialization='From scratch seed42; identical network initialization and RGB crops/order. Added recycling RNG draws alter subsequent timestep/noise draws; time strata stay50/50.',
                    recycling=dict(fraction=0.5, selection='Exactly half each batch, random membership independent of fixed timestep positions',
                                   teacher='Same current model, no_grad, independent pure Gaussian noise at t999, clamp x0 to[-1,1]',
                                   second_input='q_sample(predicted_x0,t,fresh independent Gaussian); other half use GT q_sample',
                                   supervision='Original GT classes and original +/-1 x0; unchanged0.5CE+0.5foregroundDice+1.0MSE',
                                   compute='One additional no_grad sample forward per batch2,2400 teacher sample forwards. Same updates is not same compute.'),
                    queue=['preflight/tests/resume', 'baseline1step_seed3407', 'baseline1step_seed3408',
                           'baseline5step_seed3407', 'baseline5step_seed3408', 'one_recycling_P1', 'candidate20step_seed3408', 'summarize_and_stop'],
                    checkpoint_selection=dict(split='val', images=25, steps=20, sampling_seed=3407, every_epochs=10, metric='miou_4class'),
                    reporting='Compare both sampling seeds to high-noise baseline; directional screen requires mIoU and monolayer IoU improve at both seeds. Report all effects, no significance claim or guaranteed gain.',
                    scope='Exactly one40-epoch/2400-update recycling P1, four baseline1/5-step full-val evaluations, one candidate3408 full-val evaluation. No dual heads, class/loss/architecture changes, other training seed, formal run or test.')
        pp = BATCH / 'plan.json'
        if pp.exists() and json.loads(pp.read_text()) != plan:
            raise RuntimeError('Frozen plan, source or protected baseline differs')
        write_json(pp, plan)
        verify(plan)
        old_status = BATCH / 'status.json'
        if old_status.exists() and json.loads(old_status.read_text())['status'] == 'completed':
            print('BATCH_ALREADY_COMPLETED', flush=True)
            return
        status('running')
        if not (BATCH / 'preflight_passed.json').exists():
            command(ROOT / 'scripts/check_environment.py', '--output', BATCH / 'environment.json')
            with (BATCH / 'storage_preflight.json').open('w') as log:
                subprocess.run([sys.executable, 'scripts/check_storage.py', '--runs', '1', '--parameters', '5132676'],
                               cwd=ROOT, stdout=log, check=True)
            with (BATCH / 'unit_tests.log').open('w') as log:
                subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not (BATCH / 'resume_verification.json').exists():
                command(ROOT / 'work/verify_p1_recycling_resume.py', '--output', BATCH / 'resume_verification.json')
            assert json.loads((BATCH / 'resume_verification.json').read_text())['status'] == 'passed'
            verify(plan)
            write_json(BATCH / 'preflight_passed.json', dict(status='passed', source_sha256=frozen))
        sweep = {}
        for steps in [1, 5]:
            for seed in [3407, 3408]:
                STAGE = f'baseline_steps{steps}_seed{seed}'
                status('running')
                sweep[STAGE] = evaluate(BASE, STAGE, steps, seed, plan)
        # 20-step measurements are reused as recorded, with their historical evaluation source.
        sweep['baseline_steps20_seed3407'] = json.loads((BASE / 'best_val_metrics.json').read_text())
        sweep['baseline_steps20_seed3408'] = previous_result
        write_json(BATCH / 'baseline_step_sweep.json', sweep)
        STAGE = 'training_recycling'
        verify(plan)
        status('running')
        if RUN.exists():
            assert json.loads((RUN / 'config.json').read_text()) == config
            assert json.loads((RUN / 'run_metadata.json').read_text())['source_sha256'] == frozen
            if json.loads((RUN / 'status.json').read_text())['status'] != 'completed':
                command(ROOT / 'scripts/train.py', '--resume', RUN / 'checkpoints/last.pt')
        else:
            command(ROOT / 'scripts/train.py', '--config', ROOT / 'configs/pilot_recycling.json', '--run-id', BATCH_ID)
        rs = json.loads((RUN / 'status.json').read_text())
        assert rs['status'] == 'completed' and rs['last_completed_epoch'] == 40 and rs['global_step'] == 2400
        rows = list(csv.DictReader((RUN / 'metrics_epochs.csv').open()))
        def values(tag):
            return [float(r['value']) for r in rows if r['tag'] == tag]
        high = values('diffusion_train/high_noise_fraction')
        rec = values('diffusion_train/recycle_fraction')
        teacher = values('diffusion_train/teacher_sample_forwards')
        assert len(high) == len(rec) == len(teacher) == 40
        assert all(v == 0.5 for v in high + rec) and sum(teacher) == 2400
        STAGE = 'candidate_seed3408'
        status('running')
        candidate3408 = evaluate(RUN, 'candidate_seed3408', 20, 3408, plan)
        baseline3407 = json.loads((BASE / 'best_val_metrics.json').read_text())
        candidate3407 = json.loads((RUN / 'best_val_metrics.json').read_text())
        pairs = {'3407': (baseline3407['metrics'], candidate3407['metrics']),
                 '3408': (previous_result['metrics'], candidate3408['metrics'])}
        keys = ['miou_4class', 'miou_foreground', 'iou_monolayer', 'iou_fewlayer', 'boundary_f1_foreground', 'seconds']
        deltas = {seed: {k: new[k] - old[k] for k in keys} for seed, (old, new) in pairs.items()}
        screen = all(deltas[s]['miou_4class'] > 0 and deltas[s]['iou_monolayer'] > 0 for s in pairs)
        report = dict(baseline_seed3407=baseline3407, candidate_seed3407=candidate3407,
                      baseline_seed3408=previous_result, candidate_seed3408=candidate3408,
                      deltas=deltas, directional_screen_passed=screen, baseline_step_sweep=sweep,
                      actual_high_noise_samples=2400, actual_recycled_samples=2400,
                      additional_teacher_sample_forwards=int(sum(teacher)), optimizer_steps=2400,
                      source_sha256=frozen, scope=plan['scope'],
                      interpretation='One training seed, two sampling seeds. Best selected only at3407. Extra training compute is reported; no significance claim.')
        verify(plan)
        write_json(BATCH / 'comparison_summary.json', report)
        STAGE = 'finished'
        status('completed', directional_screen_passed=screen)
    except BaseException as exc:
        status('failed', error=str(exc))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

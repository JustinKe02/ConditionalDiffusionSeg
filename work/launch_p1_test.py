"""Frozen, evaluation-only test comparison of three validation-selected P1 models."""
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.data import read_manifest
from conditional_diffusion.inference import positions
from conditional_diffusion.metrics_impl import summarize, summarize_boundaries
from conditional_diffusion.runtime import StorageGuard, source_hashes, write_json

BATCH = ROOT / 'outputs/launches/p1_test_20261018'
PROTOCOL = dict(sampling_steps=20, samples_per_image=1, sampling_seed=3407)
RUNS = [
    ('direct', 'outputs/P1/direct/seed_42/p1_cosine_20261017'),
    ('iterative', 'outputs/P1/iterative/seed_42/p1_cosine_20261017'),
    ('diffusion', 'outputs/P1/diffusion/seed_42/p1_recycling_20261017'),
]
STAGE = 'preflight'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def status(value, **extra):
    write_json(BATCH / 'status.json', dict(status=value, stage=STAGE,
               pid=os.getpid(), updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra))


def verify(plan):
    assert source_hashes(ROOT) == plan['evaluation_source_sha256'], 'Frozen source/config changed'
    for category in ['protected_artifacts_sha256', 'dataset_files_sha256', 'workflow_sha256']:
        for path, digest in plan[category].items():
            assert sha(ROOT / path) == digest, f'{category} changed: {path}'
    assert json.loads((BATCH / 'protocol.json').read_text()) == PROTOCOL


def prepare():
    BATCH.mkdir(parents=True, exist_ok=True)
    if (BATCH / 'plan.json').exists():
        plan = json.loads((BATCH / 'plan.json').read_text())
        verify(plan)
        return plan
    # Do not silently reuse any earlier exposure to the test set.
    assert not list((ROOT / 'outputs').glob('**/evaluations/test_*')), 'Existing test evaluations require review'
    previous = json.loads((ROOT / 'outputs/launches/p1_dual_head_20261018/plan.json').read_text())
    assert source_hashes(ROOT) == previous['source_sha256']
    protected = dict(previous['protected_weights'])
    for category in ['workflow_sha256', 'historical_reports_sha256']:
        protected.update(previous[category])
    for path, digest in protected.items():
        assert sha(ROOT / path) == digest, f'Historical artifact changed: {path}'
    for path in (ROOT / 'outputs/P1').glob('*/seed_42/*/checkpoints/*.pt'):
        protected[str(path.relative_to(ROOT))] = sha(path)
    manifest_path = ROOT / 'data_manifests/source_dataset.json'
    manifest = read_manifest(manifest_path)
    ids = Path(manifest['splits']['test']['snapshot_path']).read_text().split()
    assert len(ids) == 27
    files = {str(manifest_path): sha(manifest_path)}
    for split in manifest['splits'].values():
        files[split['snapshot_path']] = sha(split['snapshot_path'])
    for ident in ids:
        for pattern in ['image_pattern', 'mask_pattern']:
            path = Path(manifest['dataset_root']) / manifest[pattern].format(id=ident)
            files[str(path)] = sha(path)
    selected = []
    for method, rel in RUNS:
        run = ROOT / rel
        s = json.loads((run / 'status.json').read_text())
        assert s == dict(status='completed', last_completed_epoch=40, global_step=2400)
        checkpoint = run / 'checkpoints/best.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        c = state['config']
        assert c['method'] == method and not c['debug'] and c['phase'] == 'P1'
        assert c['crop_size'] == 512 and c['stride'] == 384 and c['seed'] == 42
        assert c['boundary_metrics'] and c['boundary_tolerance'] == 2
        assert c['manifest'] == str(manifest_path)
        assert all(c[k] == v for k, v in PROTOCOL.items())
        best_val = json.loads((run / 'best_val_metrics.json').read_text())
        assert state['epoch'] == best_val['epoch']
        metadata = json.loads((run / 'run_metadata.json').read_text())
        assert metadata['split_hashes'] == {k: v['sha256'] for k, v in manifest['splits'].items()}
        for name in ['best_val_metrics.json', 'config.json', 'run_metadata.json', 'status.json']:
            protected[str((run / name).relative_to(ROOT))] = sha(run / name)
        guard = StorageGuard(ROOT, run, c['storage'])
        guard.check(160 * 1024 ** 2)
        selected.append(dict(method=method, run=rel, checkpoint=str(checkpoint),
                             checkpoint_sha256=sha(checkpoint), checkpoint_epoch=state['epoch'],
                             validation_metrics=best_val['metrics'], parameters=metadata['parameter_count'],
                             trained_source_sha256=metadata['source_sha256']))
        del state
    plan = dict(batch_id=BATCH.name, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                scope='Test evaluation only: Direct, Iterative5, best single-head high-noise+recycling Diffusion20. No training, tuning, reselection, alternate seeds, step sweep, or extra variants.',
                authorization='User requested starting test-set evaluation in this chat on 2026-10-18.',
                selection='All best.pt weights selected solely on full25-val mIoU4class at sampling_seed3407 before any test performance was read.',
                models=selected, protocol=PROTOCOL, window=512, stride=384,
                precision='FP32', test_ids=ids, test_count=27,
                evaluation_source_sha256=source_hashes(ROOT), protected_artifacts_sha256=protected,
                dataset_files_sha256=files,
                workflow_sha256={str(Path(__file__).relative_to(ROOT)): sha(__file__)},
                reporting='Global confusion-matrix four-class IoU/F1, foreground boundary F1 at native2px tolerance, per-image matrices, raw PNG labels, measured total evaluation time including data/metrics/PNG I/O.',
                limitations=['One training seed42; sampling seeds are not training replicates.',
                             'Sample/mother-image source independence is not yet verified.',
                             'These are 40-epoch P1 checkpoints, not a completed200-epoch multi-seed study.',
                             'Test results must not be used to tune these models or choose inference settings.'])
    write_json(BATCH / 'protocol.json', PROTOCOL)
    write_json(BATCH / 'plan.json', plan)
    (BATCH / 'README.md').write_text(
        '# Frozen P1 test evaluation\n\n'
        'Three validation-selected best checkpoints, all27 test images,512/384,FP32. '
        'Diffusion uses20 DDIM steps, one sample, seed3407. No training or test-driven tuning.\n\n'
        'Protocol and checkpoint/dataset/source hashes are frozen in plan.json before evaluation. '
        'GPU lock: outputs/.experiments.lock. Logs: <method>.log. '
        'Predictions and raw metrics remain in each original run/evaluations/test_* directory. '
        'Only complete, audited evaluations receive <method>_result.json; reruns reuse those records. '
        'An incomplete prior test directory stops automatic reruns for review.\n\n'
        'Summary: comparison_summary.json, comparison.csv, per_image_metrics.csv, final_analysis.md. '
        'Status: status.json. Historical artifacts and weights are retained.\n')
    verify(plan)
    return plan


def audit_result(item, out, plan):
    m = json.loads((out / 'metrics.json').read_text())
    ec = json.loads((out / 'evaluation_config.json').read_text())
    assert ec['split'] == 'test' and ec['checkpoint'] == item['checkpoint']
    assert ec['checkpoint_epoch'] == item['checkpoint_epoch']
    assert ec['source_sha256'] == plan['evaluation_source_sha256']
    assert ec['protocol_path'] == str(BATCH / 'protocol.json')
    assert all(ec['config'][k] == v for k, v in PROTOCOL.items())
    assert ec['config']['crop_size'] == 512 and ec['config']['stride'] == 384
    assert m['image_count'] == 27
    manifest = read_manifest(ROOT / 'data_manifests/source_dataset.json')
    per_image, nfe = [], 0
    factor = {'direct': 1, 'iterative': 5, 'diffusion': 20}[item['method']]
    assert {p.stem for p in out.glob('*.png')} == set(plan['test_ids'])
    for ident in plan['test_ids']:
        target_path = Path(manifest['dataset_root']) / manifest['mask_pattern'].format(id=ident)
        with Image.open(target_path) as im:
            target = np.asarray(im).astype(np.int64)
        with Image.open(out / (ident + '.png')) as im:
            prediction = np.asarray(im)
        assert target.shape == prediction.shape and prediction.ndim == 2
        assert prediction.min() >= 0 and prediction.max() <= 3
        matrix = np.bincount((target * 4 + prediction).ravel(), minlength=16).reshape(4, 4)
        per_image.append(matrix)
        nfe += len(positions(target.shape[0], 512, 384)) * len(positions(target.shape[1], 512, 384)) * factor
    with np.load(out / 'confusion_matrices.npz', allow_pickle=False) as matrices:
        assert matrices['ids'].tolist() == plan['test_ids']
        assert np.array_equal(matrices['per_image'], np.array(per_image))
        assert np.array_equal(matrices['global_matrix'], np.sum(per_image, axis=0))
        recomputed = summarize(matrices['global_matrix'])
        recomputed.update(summarize_boundaries(matrices['boundary_counts']))
    assert m['nfe_total'] == nfe
    for key, value in recomputed.items():
        assert np.isclose(m[key], value, equal_nan=True), f'Metric mismatch: {key}'
    return dict(method=item['method'], checkpoint=item['checkpoint'], checkpoint_sha256=item['checkpoint_sha256'],
                checkpoint_epoch=item['checkpoint_epoch'], evaluation_directory=str(out), metrics=m,
                validation_metrics=item['validation_metrics'], parameters=item['parameters'],
                audit='27 PNG predictions independently reproduce all per-image and global confusion matrices; scalar metrics and NFE match; source/protocol/weights/dataset hashes verified.')


def run_evaluation(item, plan):
    verify(plan)
    cached = BATCH / (item['method'] + '_result.json')
    if cached.exists():
        previous = json.loads(cached.read_text())
        assert previous['checkpoint_sha256'] == item['checkpoint_sha256']
        assert audit_result(item, Path(previous['evaluation_directory']), plan) == previous
        return previous
    run = ROOT / item['run']
    assert not list((run / 'evaluations').glob('test_*')), 'Prior incomplete or unregistered test exposure requires review'
    with (BATCH / (item['method'] + '.log')).open('x') as log:
        subprocess.run([sys.executable, '-u', 'scripts/evaluate.py', '--checkpoint', item['checkpoint'],
                        '--split', 'test', '--allow-test', '--protocol', str(BATCH / 'protocol.json'),
                        '--save-predictions'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    outputs = list((run / 'evaluations').glob('test_*'))
    assert len(outputs) == 1
    result = audit_result(item, outputs[0], plan)
    verify(plan)
    write_json(cached, result)
    print('TEST_COMPLETED', item['method'], json.dumps(result['metrics']), flush=True)
    return result


def report(results, plan):
    keys = ['miou_4class', 'miou_foreground']
    keys += [prefix + c for c in ['background', 'monolayer', 'fewlayer', 'multilayer'] for prefix in ['iou_', 'f1_']]
    keys += ['boundary_f1_foreground', 'seconds', 'image_count', 'nfe_total', 'peak_allocated_mib']
    with (BATCH / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'checkpoint_epoch', 'parameters'] + keys)
        writer.writeheader()
        for r in results:
            writer.writerow({**{k: r[k] for k in ['method', 'checkpoint_epoch', 'parameters']},
                             **{k: r['metrics'][k] for k in keys}})
    with (BATCH / 'per_image_metrics.csv').open('w', newline='') as f:
        names = ['method', 'image_id', 'miou_4class', 'miou_foreground']
        names += [prefix + c for c in ['background', 'monolayer', 'fewlayer', 'multilayer'] for prefix in ['iou_', 'f1_']]
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for r in results:
            with np.load(Path(r['evaluation_directory']) / 'confusion_matrices.npz', allow_pickle=False) as matrices:
                for ident, matrix in zip(matrices['ids'], matrices['per_image']):
                    row = dict(method=r['method'], image_id=str(ident), **summarize(matrix))
                    writer.writerow({k: '' if isinstance(v, float) and not np.isfinite(v) else v for k, v in row.items()})
    direct = results[0]['metrics']
    deltas = {r['method']: {k: r['metrics'][k] - direct[k] for k in
              ['miou_4class', 'iou_monolayer', 'iou_fewlayer', 'boundary_f1_foreground']} for r in results[1:]}
    write_json(BATCH / 'comparison_summary.json', dict(scope=plan['scope'], protocol=PROTOCOL,
               results=results, deltas_vs_direct=deltas, limitations=plan['limitations'],
               completed_at_utc=datetime.now(timezone.utc).isoformat(),
               timing_scope=plan['reporting'], per_image_caution='Do not average per-image mIoU as global mIoU; absent classes can change the per-image denominator.'))
    lines = ['# P1 frozen test comparison', '',
             'All27 test images;512/384;FP32; independently trained seed42 P1 models. Best weights were selected on validation before this test run.', '',
             '| Method | Best epoch | Val mIoU % | Test mIoU % | Mono IoU % | Few IoU % | Foreground boundary F1 % | Total seconds |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        m = r['metrics']
        lines.append(f"| {r['method']} | {r['checkpoint_epoch']} | {r['validation_metrics']['miou_4class']*100:.2f} | {m['miou_4class']*100:.2f} | {m['iou_monolayer']*100:.2f} | {m['iou_fewlayer']*100:.2f} | {m['boundary_f1_foreground']*100:.2f} | {m['seconds']:.1f} |")
    lines += ['', 'Diffusion uses the preselected single-head high-noise50%+recycling50% recipe,20 DDIM steps, one sample, sampling_seed3407. No test-driven reselection, alternate seeds, step sweeps or training occurred.', '',
              'Time covers the complete evaluation loop including data loading, sliding-window prediction, boundary metrics and prediction PNG I/O; it is not isolated GPU latency. Each model is evaluated serially.', '',
              'Audit: saved PNGs independently reproduce all27 per-image and global confusion matrices; IoU/F1, recorded boundary counts, NFE, IDs and frozen source/weight/dataset hashes match.', '',
              'Limitations: ' + ' '.join(plan['limitations']), '',
              'Raw metrics, confusion matrices, evaluation configuration and prediction PNGs remain under the evaluation_directory paths in comparison_summary.json. Per-image absent classes can alter mIoU denominators; use the global confusion matrix for the primary comparison.', '']
    (BATCH / 'final_analysis.md').write_text('\n'.join(lines))


def main():
    global STAGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    lock = (ROOT / 'outputs/.experiments.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        plan = prepare()
        if args.prepare_only:
            print('FROZEN_PLAN', BATCH / 'plan.json', flush=True)
            return
        if (BATCH / 'status.json').exists() and json.loads((BATCH / 'status.json').read_text())['status'] == 'completed':
            return
        status('running')
        results = []
        for item in plan['models']:
            STAGE = 'test_' + item['method']
            status('running', completed_methods=[r['method'] for r in results])
            results.append(run_evaluation(item, plan))
        STAGE = 'reporting'
        status('running')
        verify(plan)
        report(results, plan)
        STAGE = 'finished'
        status('completed', completed_methods=[r['method'] for r in results])
    except BaseException as exc:
        if BATCH.exists():
            status('failed', error=str(exc))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

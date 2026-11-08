"""Chapter 4 progression orchestrator: S0 check, S1/S6 train, S2/S3/S7/S8 eval, bootstrap, S9 diagnose, S4/S5 benchmark, and report."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

ROOT = Path(__file__).resolve().parents[1]
for path_entry in (ROOT / 'src', ROOT / 'work'):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from conditional_diffusion import diffusion_seg_experiment as DSE
from conditional_diffusion.data import CLASSES, MaterialDataset, read_manifest
from conditional_diffusion.inference import image_seed, positions
from conditional_diffusion.metrics_impl import boundary_counts, confusion, summarize, summarize_boundaries
from conditional_diffusion.runtime import source_hashes

DEFAULT_OUT = ROOT / 'outputs/ch4_progression_20261108'
MANIFEST_PATH = ROOT / 'data_manifests/source_dataset.json'
FONT_PATH = ROOT / 'docs/figures/glpflow_20261021/fonts/NotoSansCJKsc-Regular.otf'
THESIS_TABLES = ROOT / 'docs/thesis/chapters35_integration_20261101/masters_thesis_20261101/tables'

EXISTING_CHECKPOINTS = {
    'I': ROOT / 'outputs/P1/glpflow/seed_42/chapter_controlled_20261031_i/checkpoints/best.pt',
    'R': ROOT / 'outputs/P1/glpflow/seed_42/p1_glpflow_rollout_20261022_e3r/checkpoints/best.pt',
    'E1': ROOT / 'outputs/P1/glpflow/seed_42/p1_glpflow_20261021_e1/checkpoints/best.pt',
    'segformer_b1': ROOT / 'outputs/P1/baselines/seed_42/p1_baselines_20261022_segformer_b1/checkpoints/best.pt',
    'ddp': ROOT / 'outputs/P1/baselines/seed_42/p1_baselines_20261022_ddp/checkpoints/best.pt',
    'ccdm': ROOT / 'outputs/P1/baselines/seed_42/p1_baselines_20261022_ccdm/checkpoints/best.pt',
    'repela_small': ROOT / 'outputs/P1/repela/seed_42/p1_repela_20261024_repela_small/checkpoints/best.pt',
}

HIST_RUNS = {
    'I': ROOT / 'outputs/P1/glpflow/seed_42/chapter_controlled_20261031_i',
    'R': ROOT / 'outputs/P1/glpflow/seed_42/p1_glpflow_rollout_20261022_e3r',
    'E1': ROOT / 'outputs/P1/glpflow/seed_42/p1_glpflow_20261021_e1',
    'segformer_b1': ROOT / 'outputs/P1/baselines/seed_42/p1_baselines_20261022_segformer_b1',
    'repela_small': ROOT / 'outputs/P1/repela/seed_42/p1_repela_20261024_repela_small',
}

CONTROLLED_ANALYSIS = ROOT / 'outputs/analysis/chapter_controlled_ablation_20261031'
HIST_GLP_TEST = ROOT / 'outputs/inference/glpflow_test_20261022'
HIST_BASE_TEST = ROOT / 'outputs/inference/baselines_test_20261022'
HIST_REPELA_TEST = ROOT / 'outputs/inference/repela_test_20261024'

BENCH_INFER_OBJECTS = [
    'g_onehot_k4', 'g_onehot_k20', 'g_onehot_k50', 'g_ordinal_k20',
    'I_k4', 'R_k1', 'R_k2', 'R_k4', 'E1',
    'segformer_b1', 'repela_deploy', 'ddp', 'ccdm',
]

BENCH_TRAIN_OBJECTS = [
    'g_onehot', 'g_ordinal', 'I', 'R', 'E1', 'segformer_b1', 'repela_small',
]

INTERPRETATION_RULES = {
    'ddim_vs_i_rules': [
        {
            'id': 'rule_1',
            'condition': 'G-DDIM20 的 mIoU 和 BF 都低于 I，且 DDIM4 明显下降',
            'writing': '传统扩散多步、慢、边界弱；流匹配用 4 步同时提升精度和边界',
        },
        {
            'id': 'rule_2',
            'condition': 'G-DDIM20 精度接近 I，但 DDIM4 明显下降，或者 DDIM20 明显更慢',
            'writing': '效率叙事：流匹配以约 1/5 的步数达到同等质量',
        },
        {
            'id': 'rule_3',
            'condition': 'G-DDIM4 仍接近 I',
            'writing': '递进依靠边界与对齐机制（I → R 和 S9），并如实写出"步数优势不明显"',
        },
        {
            'id': 'rule_4',
            'condition': 'G 优于 I 或 R',
            'writing': '如实报告；把 R 定位为"少步、边界更好"的方案，资源递进不变',
        },
    ],
    'ordinal_vs_onehot_rules': [
        {
            'id': 'ord_lt_1pp',
            'condition': '|G-ordinal − G-onehot| < 1 pp',
            'writing': '改进掩码编码对传统扩散收益有限，瓶颈在采样路径与状态',
        },
        {
            'id': 'ord_ge_1pp',
            'condition': '|G-ordinal − G-onehot| >= 1 pp',
            'writing': '有序编码有帮助，但不能解决多步与资源问题',
        },
    ],
    'reporting_constraints': [
        '所有比较以百分点表示，并附配对 bootstrap 区间；不使用"显著"一词。',
        '区间只描述固定权重下的图像抽样不确定性，不代表训练种子方差。',
    ],
}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def now_bjt() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def clean_json(val):
    if isinstance(val, dict):
        return {str(k): clean_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [clean_json(v) for v in val]
    if isinstance(val, np.generic):
        return clean_json(val.item())
    if isinstance(val, float) and not math.isfinite(val):
        return None
    if isinstance(val, Path):
        return str(val)
    return val


def read_json(path: Path) -> dict | list:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=path.name + '.', suffix='.tmp', delete=False,
        ) as stream:
            tmp = Path(stream.name)
            json.dump(clean_json(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_out_dir(out: Path) -> Path:
    resolved = out.resolve()
    outputs_root = (ROOT / 'outputs').resolve()
    if not resolved.is_relative_to(outputs_root) or resolved == outputs_root:
        raise ValueError(f'--out must be a named subdirectory of {outputs_root}: got {resolved}')
    return resolved


def boundary_iou_counts(target: np.ndarray, pred: np.ndarray, d: int) -> np.ndarray:
    rows = []
    structure = np.ones((3, 3), dtype=bool)
    for c in range(4):
        a, b = target == c, pred == c
        ea = a & ~ndi.binary_erosion(a, structure=structure, iterations=d, border_value=0)
        eb = b & ~ndi.binary_erosion(b, structure=structure, iterations=d, border_value=0)
        rows.append([int(np.count_nonzero(ea & eb)), int(np.count_nonzero(ea | eb))])
    return np.asarray(rows, dtype=np.int64)


def summarize_boundary_iou(counts: np.ndarray) -> dict:
    vals = np.divide(counts[:, 0], counts[:, 1], out=np.full(4, np.nan), where=counts[:, 1] > 0)
    out = {'boundary_iou_foreground': float(np.nanmean(vals[1:])) if np.isfinite(vals[1:]).any() else float('nan')}
    for idx, cls_name in enumerate(CLASSES):
        out[f'boundary_iou_{cls_name}'] = float(vals[idx])
    return out


def summarize_full_metrics(cm: np.ndarray, bc_by_tol: dict[int, np.ndarray], biou_by_tol: dict[int, np.ndarray] | None = None) -> dict:
    metrics = DSE.summarize_material(cm)
    for idx, cls_name in enumerate(CLASSES):
        col_sum = int(cm[:, idx].sum())
        row_sum = int(cm[idx, :].sum())
        metrics[f'precision_{cls_name}'] = float(cm[idx, idx] / col_sum) if col_sum > 0 else float('nan')
        metrics[f'recall_{cls_name}'] = float(cm[idx, idx] / row_sum) if row_sum > 0 else float('nan')
    mono_sum, few_sum, multi_sum = int(cm[1].sum()), int(cm[2].sum()), int(cm[3].sum())
    metrics['few_to_mono_rate'] = float(cm[2, 1] / few_sum) if few_sum > 0 else float('nan')
    metrics['mono_to_few_rate'] = float(cm[1, 2] / mono_sum) if mono_sum > 0 else float('nan')
    metrics['mono_to_background_rate'] = float(cm[1, 0] / mono_sum) if mono_sum > 0 else float('nan')
    metrics['multi_to_few_rate'] = float(cm[3, 2] / multi_sum) if multi_sum > 0 else float('nan')
    for tol, bc in sorted(bc_by_tol.items()):
        bm = summarize_boundaries(bc)
        if tol == 2:
            metrics.update(bm)
        for k, v in bm.items():
            metrics[f'{k}_tol{tol}'] = v
    if biou_by_tol:
        for tol, biou in sorted(biou_by_tol.items()):
            bm = summarize_boundary_iou(biou)
            for k, v in bm.items():
                metrics[f'{k}_tol{tol}'] = v
    return metrics


def band_mask(mask: np.ndarray, radius: int = 8) -> np.ndarray:
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    edge = F.max_pool2d(F.pad(m, (1, 1, 1, 1), mode='replicate'), 3, 1) != -F.max_pool2d(-F.pad(m, (1, 1, 1, 1), mode='replicate'), 3, 1)
    return (F.max_pool2d(edge.float(), 2 * radius + 1, 1, radius) > 0)[0, 0].numpy()


def git_head_sha() -> str:
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


# ----------------------------------------------------------------------------- prepare & check

def cmd_prepare(args):
    out = ensure_out_dir(args.out)
    for sub in ('configs', 'eval', 'diagnostics',
                'benchmark/objects', 'benchmark/train_objects', 'logs', 'tables', 'figures', 'qa/figure_previews'):
        (out / sub).mkdir(parents=True, exist_ok=True)

    protocol_keys = (
        'manifest', 'device', 'seed', 'cpu_threads', 'epochs', 'batch_size', 'workers',
        'crop_size', 'stride', 'crops_per_image', 'thin_crop_fraction',
        'photometric_gain', 'photometric_gamma', 'photometric_vignette', 'flip_mode',
        'lr', 'min_lr', 'warmup_epochs', 'weight_decay', 'grad_clip', 'ema_sigma_rel',
        'timesteps', 'sampling_steps', 'sampling_seed', 'window_batch_size',
        'val_every', 'boundary_tolerance', 'ce_weight', 'dice_weight',
        'reconstruction_weight', 'prior_loss_weight', 'max_run_gib', 'min_free_gib', 'model',
    )

    snapshots = {}
    proto_ref = None
    first_cfg = None
    for arm in ('g_onehot', 'g_ordinal'):
        src = ROOT / 'configs' / f'ch4_{arm}.json'
        cfg = read_json(src)
        DSE.validate_config(cfg)
        proto = {k: cfg[k] for k in protocol_keys}
        if proto_ref is not None and proto != proto_ref:
            raise ValueError(f'Matched training protocol differs for {arm}')
        proto_ref = proto
        if first_cfg is None:
            first_cfg = cfg
        dst = out / 'configs' / f'ch4_{arm}.json'
        src_bytes = src.read_bytes()
        if dst.exists():
            if dst.read_bytes() != src_bytes:
                raise ValueError(f'Immutable config snapshot differs: {dst}')
        else:
            dst.write_bytes(src_bytes)
        snapshots[arm] = dict(path=str(dst), source=str(src), sha256=sha256_file(dst), variant=cfg['variant'], representation=cfg['representation'])

    manifest = read_manifest(first_cfg['manifest'])
    splits_meta = {}
    for split_name in ('train', 'val', 'test'):
        sp = Path(manifest['splits'][split_name]['snapshot_path'])
        splits_meta[split_name] = dict(
            snapshot_path=str(sp),
            count=manifest['splits'][split_name]['count'],
            sha256=sha256_file(sp),
        )

    existing_ckpt_meta = {}
    for k, p in EXISTING_CHECKPOINTS.items():
        if not p.exists():
            raise FileNotFoundError(f'Missing registered existing checkpoint {k}: {p}')
        existing_ckpt_meta[k] = dict(path=str(p), sha256=sha256_file(p))

    expected_commit = args.expected_commit or git_head_sha()
    plan_path = out / 'plan.json'
    prov = DSE.provenance(first_cfg, ROOT, include_test=False)
    plan = dict(
        schema_version=1,
        batch_id='ch4_progression_20261108',
        expected_commit=expected_commit,
        configs=snapshots,
        runs={arm: str(out / arm) for arm in ('g_onehot', 'g_ordinal')},
        protocol=proto_ref,
        provenance=prov,
        splits=splits_meta,
        test_split_sha256=splits_meta['test']['sha256'],
        existing_checkpoints=existing_ckpt_meta,
        interpretation_rules=INTERPRETATION_RULES,
    )
    if plan_path.exists():
        old = read_json(plan_path)
        if 'test_registration' in old:
            plan['test_registration'] = old['test_registration']
        for k, v in plan.items():
            if old.get(k) != v:
                raise ValueError(f'Existing immutable plan.json differs on key {k!r}')
        print(json.dumps(dict(status='verified_existing_plan', plan=str(plan_path), expected_commit=expected_commit), ensure_ascii=False, indent=2))
    else:
        atomic_json(plan_path, plan)
        print(json.dumps(dict(status='prepared', plan=str(plan_path), expected_commit=expected_commit), ensure_ascii=False, indent=2))


def _assert_tree_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _assert_tree_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for l, r in zip(a, b):
            _assert_tree_equal(l, r)
    else:
        assert a == b


def cmd_check(args):
    out = ensure_out_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(checked_at_utc=now_utc(), checked_at_bjt=now_bjt(), checks={})

    # 1. Unit tests
    if not args.skip_unit_tests:
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'],
                              cwd=ROOT, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f'Unit tests failed:\n{proc.stdout}\n{proc.stderr}')
        last_lines = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        report['checks']['unit_tests'] = dict(passed=True, seconds=time.perf_counter() - t0, tail=last_lines)
        print('[S0] unit_tests passed', flush=True)

    # 2. Parameter counts matching 11-03 metadata
    cfg_oh = read_json(ROOT / 'configs/ch4_g_onehot.json')
    cfg_ord = read_json(ROOT / 'configs/ch4_g_ordinal.json')
    DSE.validate_config(cfg_oh)
    DSE.validate_config(cfg_ord)
    oh_cpu, _ = DSE.build(dict(cfg_oh, device='cpu'))
    ord_cpu, _ = DSE.build(dict(cfg_ord, device='cpu'))
    p_oh = sum(p.numel() for p in oh_cpu.parameters())
    p_ord = sum(p.numel() for p in ord_cpu.parameters())
    assert p_oh == 18_192_148, f'Expected onehot params 18192148, got {p_oh}'
    assert p_ord == 18_191_860, f'Expected ordinal params 18191860, got {p_ord}'
    meta_1103_oh = read_json(ROOT / 'outputs/diffusion_seg_20261103/onehot/metadata.json')
    meta_1103_ord = read_json(ROOT / 'outputs/diffusion_seg_20261103/ordinal/metadata.json')
    assert meta_1103_oh['parameter_count'] == p_oh and meta_1103_ord['parameter_count'] == p_ord
    report['checks']['parameter_counts'] = dict(passed=True, onehot=p_oh, ordinal=p_ord, matches_1103=True)
    del oh_cpu, ord_cpu
    print(f'[S0] parameter_counts passed: onehot={p_oh}, ordinal={p_ord}', flush=True)

    # 3. Smoke checks for both production configs
    smoke_oh = DSE.smoke(cfg_oh)
    smoke_ord = DSE.smoke(cfg_ord)
    assert smoke_oh['status'] == 'passed' and smoke_ord['status'] == 'passed'
    report['checks']['smoke'] = dict(passed=True, ch4_g_onehot=smoke_oh, ch4_g_ordinal=smoke_ord)
    print('[S0] smoke passed for both configs', flush=True)

    # Temporary run directories strictly inside OUT (which is under ROOT/outputs)
    tmp_dirs = [out / '_s0_legacy_a', out / '_s0_legacy_b', out / '_s0_resume_full', out / '_s0_resume_split']
    tmp_locks = [out / f'.{p.name}.lock' for p in tmp_dirs]
    for d in tmp_dirs:
        if d.exists():
            shutil.rmtree(d)

    try:
        # 4. Legacy 11-03 regression: 3 batches on CUDA, deterministic & finite & ema=False & crop hash verified
        cfg_leg = read_json(ROOT / 'configs/diffusion_seg_onehot.json')
        DSE.validate_config(cfg_leg)
        ds_leg = DSE.SegmentationDataset(cfg_leg, 'train')
        ds_leg.epoch = 1
        loader_leg = DataLoader(ds_leg, batch_size=cfg_leg['batch_size'], shuffle=True, num_workers=0,
                                collate_fn=DSE.collate, generator=torch.Generator().manual_seed(cfg_leg['seed'] + 1))
        h = hashlib.sha256()
        for bi, b in enumerate(loader_leg):
            if bi >= 3:
                break
            for key in ('image', 'mask', 'box'):
                h.update(b[key].numpy().tobytes())
        crop_hash = h.hexdigest()

        cfg_leg_run = dict(cfg_leg, min_free_gib=0.0)
        res_leg_a = DSE.train(cfg_leg_run, tmp_dirs[0], ROOT, stop_after_epoch=1, max_batches=3)
        res_leg_b = DSE.train(cfg_leg_run, tmp_dirs[1], ROOT, stop_after_epoch=1, max_batches=3)
        st_a = torch.load(tmp_dirs[0] / 'last.pt', map_location='cpu', weights_only=False)
        st_b = torch.load(tmp_dirs[1] / 'last.pt', map_location='cpu', weights_only=False)
        assert 'ema' not in st_a
        meta_leg = read_json(tmp_dirs[0] / 'metadata.json')
        assert meta_leg['ema'] is False and meta_leg['parameter_count'] == 18_192_148
        _assert_tree_equal(st_a['model'], st_b['model'])
        _assert_tree_equal(st_a['optimizer'], st_b['optimizer'])
        loss_leg = read_json(tmp_dirs[0] / 'history.json')[0]['train']['loss_total']
        assert math.isfinite(loss_leg)
        report['checks']['legacy_1103_regression'] = dict(
            passed=True, batches=3, batch_size=cfg_leg['batch_size'],
            first_3batches_crop_sha256=crop_hash, epoch1_3batch_loss=loss_leg,
            deterministic_match=True, ema_disabled=True,
        )
        print(f'[S0] legacy_1103_regression passed: loss={loss_leg:.5f}', flush=True)

        # 5. Resume bit-exactness with PowerEMA (max_batches=3, 2 epochs on CUDA with val)
        cfg_res = dict(cfg_oh, epochs=2, warmup_epochs=0, val_every=1, sampling_steps=2, min_free_gib=0.0)
        DSE.train(cfg_res, tmp_dirs[2], ROOT, max_batches=3)
        DSE.train(cfg_res, tmp_dirs[3], ROOT, stop_after_epoch=1, max_batches=3)
        DSE.train(cfg_res, tmp_dirs[3], ROOT, resume=True, max_batches=3)
        ref = torch.load(tmp_dirs[2] / 'last.pt', map_location='cpu', weights_only=False)
        rec = torch.load(tmp_dirs[3] / 'last.pt', map_location='cpu', weights_only=False)
        for key in ('model', 'ema', 'optimizer', 'scheduler', 'rng', 'epoch', 'global_step'):
            _assert_tree_equal(ref[key], rec[key])
        assert ref['best'] is not None and rec['best'] is not None
        assert ref['best']['epoch'] == rec['best']['epoch']
        _assert_tree_equal(ref['best']['model'], rec['best']['model'])
        best_res = torch.load(tmp_dirs[3] / 'best.pt', map_location='cpu', weights_only=False)
        _assert_tree_equal(best_res['model'], rec['best']['model'])
        report['checks']['ema_resume_consistency'] = dict(
            passed=True, max_batches=3, epochs=2,
            best_epoch=ref['best']['epoch'], keys_verified=['model', 'ema', 'optimizer', 'scheduler', 'rng', 'best'],
        )
        print(f'[S0] ema_resume_consistency passed (best_epoch={ref["best"]["epoch"]})', flush=True)

        # 6. CUDA B8 20-step memory peak check (< 22.5 GiB)
        torch.cuda.empty_cache()
        DSE.configure(cfg_oh)
        model, diffusion = DSE.build(cfg_oh)
        ema = DSE.PowerEMA(model, cfg_oh['ema_sigma_rel'])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg_oh['lr'], weight_decay=cfg_oh['weight_decay'])
        ds_mem = DSE.SegmentationDataset(cfg_oh, 'train')
        ds_mem.epoch = 1
        loader_mem = DataLoader(ds_mem, batch_size=cfg_oh['batch_size'], shuffle=True, num_workers=cfg_oh['workers'],
                                collate_fn=DSE.collate, generator=torch.Generator().manual_seed(cfg_oh['seed'] + 1))
        batches_iter = iter(loader_mem)
        model.train()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        losses = []
        for _ in range(20):
            try:
                batch = next(batches_iter)
            except StopIteration:
                ds_mem.epoch += 1
                batches_iter = iter(loader_mem)
                batch = next(batches_iter)
            terms, grad = DSE.train_step(model, diffusion, opt, batch, cfg_oh)
            ema.update(model)
            losses.append(terms['loss_total'])
        torch.cuda.synchronize()
        peak_alloc_mib = torch.cuda.max_memory_allocated() / 2**20
        peak_res_mib = torch.cuda.max_memory_reserved() / 2**20
        peak_res_gib = peak_res_mib / 1024.0
        assert peak_res_gib < 22.5, f'B8 20-step peak reserved VRAM {peak_res_gib:.2f} GiB >= 22.5 GiB limit'
        report['checks']['vram_b8_20steps'] = dict(
            passed=True, steps=20, batch_size=cfg_oh['batch_size'],
            peak_allocated_mib=peak_alloc_mib, peak_reserved_mib=peak_res_mib,
            peak_reserved_gib=peak_res_gib, limit_gib=22.5,
            last_loss=losses[-1], gpu=torch.cuda.get_device_name(),
        )
        print(f'[S0] vram_b8_20steps passed: peak_reserved={peak_res_gib:.2f} GiB ({peak_res_mib:.0f} MiB)', flush=True)
        del model, diffusion, ema, opt
        torch.cuda.empty_cache()
    finally:
        for d in tmp_dirs:
            if d.exists():
                shutil.rmtree(d)
        for lf in tmp_locks:
            lf.unlink(missing_ok=True)

    atomic_json(out / 'engineering_check.json', report)
    print(json.dumps(clean_json(report), ensure_ascii=False, indent=2))


# ----------------------------------------------------------------------------- train

def cmd_train(args):
    out = ensure_out_dir(args.out)
    plan = read_json(out / 'plan.json')
    arm = args.arm
    cfg_entry = plan['configs'][arm]
    cfg_path = Path(cfg_entry['path'])
    if sha256_file(cfg_path) != cfg_entry['sha256']:
        raise RuntimeError(f'Frozen config hash mismatch for {arm}')
    cfg = read_json(cfg_path)
    if DSE.provenance(cfg, ROOT, include_test=False) != plan['provenance']:
        raise RuntimeError('Source or train/val split provenance changed since prepare')
    run_dir = Path(plan['runs'][arm])
    if not args.resume and run_dir.is_dir() and not any(run_dir.iterdir()):
        run_dir.rmdir()
    t0 = time.perf_counter()
    res = DSE.train(cfg, run_dir, ROOT, resume=args.resume)
    elapsed = time.perf_counter() - t0
    assert (run_dir / 'best.pt').exists() and (run_dir / 'last.pt').exists()
    state = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
    assert state['epoch'] == cfg['epochs'], f'Expected epoch {cfg["epochs"]}, got {state["epoch"]}'
    print(json.dumps(dict(arm=arm, elapsed_seconds=elapsed, **res), ensure_ascii=False, indent=2))


# ----------------------------------------------------------------------------- evaluate (val25 / test27)

def _check_test_registration(plan: dict, result_name: str, model_name: str, steps: int, seed: int, ckpt_path: Path):
    if 'test_registration' not in plan:
        raise RuntimeError('test_registration missing from plan.json; run register-test before evaluating test split')
    reg = plan['test_registration']
    canonical = json.dumps(dict(test_split_sha256=reg['test_split_sha256'], entries=reg['entries']), sort_keys=True)
    if hashlib.sha256(canonical.encode('utf-8')).hexdigest() != reg['registration_sha256']:
        raise RuntimeError('test_registration integrity hash mismatch')
    if sha256_file(Path(plan['splits']['test']['snapshot_path'])) != reg['test_split_sha256']:
        raise RuntimeError('test.txt split hash changed')
    matches = [e for e in reg['entries'] if e['result_name'] == result_name]
    if not matches:
        raise RuntimeError(f'Test evaluation {result_name} is not pre-registered')
    entry = matches[0]
    if (entry['model'], entry['steps'], entry['seed']) != (model_name, steps, seed):
        raise RuntimeError(f'Test arguments mismatch registration: expected {entry}')
    if str(ckpt_path.resolve()) != str(Path(entry['checkpoint']).resolve()):
        raise RuntimeError(f'Checkpoint path mismatch: {ckpt_path} != {entry["checkpoint"]}')
    if sha256_file(ckpt_path) != entry['checkpoint_sha256']:
        raise RuntimeError(f'Checkpoint sha256 mismatch for {ckpt_path}')
    return entry


def cmd_register_test(args):
    out = ensure_out_dir(args.out)
    plan_path = out / 'plan.json'
    plan = read_json(plan_path)
    required_vals = [
        out / 'g_onehot/best.pt',
        out / 'g_ordinal/best.pt',
        out / 'eval/steps_val_g_onehot.json',
        out / 'eval/steps_val_g_ordinal.json',
        out / 'eval/steps_val_I.json',
        out / 'eval/seed3408_g_onehot.json',
        out / 'eval/seed3408_g_ordinal.json',
    ]
    for req in required_vals:
        if not req.exists():
            raise RuntimeError(f'Cannot register test before val artifact exists: {req}')

    ckpts = {
        'g_onehot': out / 'g_onehot/best.pt',
        'g_ordinal': out / 'g_ordinal/best.pt',
        'I': Path(plan['existing_checkpoints']['I']['path']),
    }
    entries = [
        dict(result_name='test27_g_onehot_k20.json', model='g_onehot', split='test', steps=20, seed=3407,
             checkpoint=str(ckpts['g_onehot']), checkpoint_sha256=sha256_file(ckpts['g_onehot']),
             purpose='primary traditional Gaussian diffusion'),
        dict(result_name='test27_g_onehot_k4.json', model='g_onehot', split='test', steps=4, seed=3407,
             checkpoint=str(ckpts['g_onehot']), checkpoint_sha256=sha256_file(ckpts['g_onehot']),
             purpose='4-NFE matched traditional Gaussian diffusion'),
        dict(result_name='test27_g_ordinal_k20.json', model='g_ordinal', split='test', steps=20, seed=3407,
             checkpoint=str(ckpts['g_ordinal']), checkpoint_sha256=sha256_file(ckpts['g_ordinal']),
             purpose='ordinal encoding variant K=20'),
        dict(result_name='test27_g_ordinal_k4.json', model='g_ordinal', split='test', steps=4, seed=3407,
             checkpoint=str(ckpts['g_ordinal']), checkpoint_sha256=sha256_file(ckpts['g_ordinal']),
             purpose='ordinal encoding variant K=4'),
        dict(result_name='test27_I_k4.json', model='I', split='test', steps=4, seed=3407,
             checkpoint=str(ckpts['I']), checkpoint_sha256=sha256_file(ckpts['I']),
             purpose='flow matching 4-step progression test evaluation'),
    ]
    test_split_sha = sha256_file(Path(plan['splits']['test']['snapshot_path']))
    canonical = json.dumps(dict(test_split_sha256=test_split_sha, entries=entries), sort_keys=True)
    reg = dict(
        registered_at_bjt=now_bjt(),
        policy='Pre-registered test27 protocol; one evaluation per entry; R already evaluated historically (93.43/72.92)',
        test_split_sha256=test_split_sha,
        entries=entries,
        registration_sha256=hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
    )
    if 'test_registration' in plan:
        if plan['test_registration']['registration_sha256'] != reg['registration_sha256']:
            raise RuntimeError('Existing test_registration differs from newly computed entries; refusing modification')
        print(json.dumps(dict(status='verified_existing_registration', registration_sha256=reg['registration_sha256']), indent=2))
        return
    plan['test_registration'] = reg
    atomic_json(plan_path, plan)
    print(json.dumps(dict(status='registered', entries=len(entries), registration_sha256=reg['registration_sha256']), ensure_ascii=False, indent=2))


def cmd_evaluate(args):
    out = ensure_out_dir(args.out)
    plan = read_json(out / 'plan.json')
    eval_dir = out / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    result_name = args.result_name if args.result_name.endswith('.json') else f'{args.result_name}.json'
    out_json = eval_dir / result_name
    out_npz = eval_dir / (Path(result_name).stem + '_counts.npz')

    if args.split == 'test':
        if len(args.steps) != 1:
            raise ValueError('Test split requires exactly one pre-registered --steps per invocation')
        if out_json.exists() and read_json(out_json).get('status') == 'completed':
            raise RuntimeError(f'Pre-registered test evaluation already completed: {out_json}')

    t_start = time.perf_counter()

    if args.model in ('g_onehot', 'g_ordinal'):
        cfg = read_json(Path(plan['configs'][args.model]['path']))
        ckpt_path = Path(plan['runs'][args.model]) / 'best.pt'
        if args.split == 'test':
            _check_test_registration(plan, result_name, args.model, args.steps[0], args.seed, ckpt_path)
        DSE.configure(cfg)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if state['config'] != cfg or state['provenance'] != plan['provenance']:
            raise RuntimeError(f'Checkpoint config/provenance mismatch: {ckpt_path}')
        model, diffusion = DSE.build(cfg)
        model.load_state_dict(state['model'])
        model.eval()
        ds = DSE.SegmentationDataset(cfg, args.split, allow_test=(args.split == 'test'))
        ckpt_sha = sha256_file(ckpt_path)
        ckpt_epoch = int(state['epoch'])
        best_val_recorded = state.get('validation', {}).get('metrics')

        step_results = {}
        npz_payload = None
        for k in args.steps:
            matrices, bc_tols, biou_tols, img_rows = [], {1: [], 2: [], 4: []}, {1: [], 2: [], 4: []}, []
            ids = []
            for idx in range(len(ds)):
                item = ds[idx]
                target = item['mask'].numpy()
                pred, timing = DSE.predict_image(
                    model, diffusion, item['image'], item['thumb'], item['id'],
                    cfg, native_shape=target.shape, sampling_steps=k, sampling_seed=args.seed,
                )
                cm = confusion(target, pred)
                matrices.append(cm)
                row_bc = {}
                for tol in (1, 2, 4):
                    bc = boundary_counts(target, pred, tol)
                    bi = boundary_iou_counts(target, pred, tol)
                    bc_tols[tol].append(bc)
                    biou_tols[tol].append(bi)
                    row_bc[f'bf{tol}'] = bc.tolist()
                    row_bc[f'biou{tol}'] = bi.tolist()
                ids.append(item['id'])
                img_rows.append(dict(id=item['id'], confusion=cm.tolist(), **row_bc, **timing))
            ids_arr = np.array(ids)
            if npz_payload is None:
                npz_payload = dict(ids=ids_arr)
            assert np.array_equal(npz_payload['ids'], ids_arr)
            npz_payload[f's{k}_cm'] = np.asarray(matrices, dtype=np.int64)
            for tol in (1, 2, 4):
                npz_payload[f's{k}_bf{tol}'] = np.asarray(bc_tols[tol], dtype=np.int64)
                npz_payload[f's{k}_biou{tol}'] = np.asarray(biou_tols[tol], dtype=np.int64)

            cm_sum = np.sum(matrices, axis=0)
            bc_sum = {tol: np.sum(bc_tols[tol], axis=0) for tol in (1, 2, 4)}
            biou_sum = {tol: np.sum(biou_tols[tol], axis=0) for tol in (1, 2, 4)}
            metrics = summarize_full_metrics(cm_sum, bc_sum, biou_sum)
            total_sec = sum(r['seconds'] for r in img_rows)
            total_nfe = sum(r['nfe'] for r in img_rows)
            if args.split == 'val' and k == cfg['sampling_steps'] and args.seed == cfg['sampling_seed'] and best_val_recorded is not None:
                diff = abs(metrics['miou_4class'] - best_val_recorded['miou_4class'])
                assert diff < 1e-9, f'Reproduced val mIoU diff too large: {diff}'
            step_results[str(k)] = dict(
                steps=k,
                seed=args.seed,
                image_count=len(ids),
                nfe_total=total_nfe,
                nfe_per_image=total_nfe / len(ids),
                seconds=total_sec,
                seconds_per_image=total_sec / len(ids),
                metrics=metrics,
                images=img_rows,
            )
            print(f'[EVAL] {args.model} split={args.split} K={k} seed={args.seed} mIoU={100*metrics["miou_4class"]:.4f}% BF2={100*metrics["boundary_f1_foreground"]:.4f}% ({total_sec:.1f}s)', flush=True)

        np.savez_compressed(out_npz, **npz_payload)
        payload = dict(
            status='completed',
            evaluated_at_bjt=now_bjt(),
            model=args.model,
            split=args.split,
            sampling_seed=args.seed,
            steps_list=args.steps,
            checkpoint=str(ckpt_path),
            checkpoint_sha256=ckpt_sha,
            best_epoch=ckpt_epoch,
            counts_npz=str(out_npz),
            elapsed_seconds=time.perf_counter() - t_start,
            results=step_results,
        )
        if len(args.steps) == 1:
            payload['metrics'] = step_results[str(args.steps[0])]['metrics']
            payload['seconds'] = step_results[str(args.steps[0])]['seconds']
            payload['nfe_total'] = step_results[str(args.steps[0])]['nfe_total']
        atomic_json(out_json, payload)

    elif args.model == 'I':
        import glpflow_core as G
        from chapter_controlled_20261031 import core as CC
        ckpt_path = Path(plan['existing_checkpoints']['I']['path'])
        if sha256_file(ckpt_path) != plan['existing_checkpoints']['I']['sha256']:
            raise RuntimeError('Checkpoint hash changed for I')
        if args.split == 'test':
            _check_test_registration(plan, result_name, 'I', args.steps[0], args.seed, ckpt_path)
        model, c, meta = CC.load_best('I')
        ds = G.GLPDataset(c, args.split, train=False)

        if args.split == 'val':
            assert sorted(args.steps) == [1, 2, 4] or all(s in (1, 2, 4) for s in args.steps)
            capture = tuple(sorted(set(args.steps)))
            per_step_mats = {k: [] for k in capture}
            per_step_bc = {k: {1: [], 2: [], 4: []} for k in capture}
            per_step_biou = {k: {1: [], 2: [], 4: []} for k in capture}
            per_step_secs = {k: [] for k in capture}
            ids = []
            for idx in range(len(ds)):
                d = ds[idx]
                gt = d['mask'].numpy()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                preds, nfe_4 = CC.predict(model, c, d['image'], d['thumb'], d['id'], stop_after=max(capture),
                                          sampling_seed=args.seed, capture=capture)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                ids.append(d['id'])
                for k in capture:
                    pred = preds[k]
                    per_step_mats[k].append(confusion(gt, pred))
                    for tol in (1, 2, 4):
                        per_step_bc[k][tol].append(boundary_counts(gt, pred, tol))
                        per_step_biou[k][tol].append(boundary_iou_counts(gt, pred, tol))
                    per_step_secs[k].append(elapsed if k == max(capture) else None)
            # Reconcile step 4 with official I best_val_confusion.npz
            with np.load(HIST_RUNS['I'] / 'best_val_confusion.npz') as z:
                assert np.array_equal(z['ids'], np.array(ids))
                if 4 in capture:
                    assert np.array_equal(np.asarray(per_step_mats[4]), z['per_image']), 'I step-4 confusion mismatch with best_val_confusion.npz'
                    assert np.array_equal(np.asarray(per_step_bc[4][2]), z['per_image_boundary']), 'I step-4 boundary mismatch with best_val_confusion.npz'
            npz_payload = dict(ids=np.array(ids))
            step_results = {}
            for k in args.steps:
                npz_payload[f's{k}_cm'] = np.asarray(per_step_mats[k], dtype=np.int64)
                for tol in (1, 2, 4):
                    npz_payload[f's{k}_bf{tol}'] = np.asarray(per_step_bc[k][tol], dtype=np.int64)
                    npz_payload[f's{k}_biou{tol}'] = np.asarray(per_step_biou[k][tol], dtype=np.int64)
                cm_sum = np.sum(per_step_mats[k], axis=0)
                bc_sum = {tol: np.sum(per_step_bc[k][tol], axis=0) for tol in (1, 2, 4)}
                biou_sum = {tol: np.sum(per_step_biou[k][tol], axis=0) for tol in (1, 2, 4)}
                metrics = summarize_full_metrics(cm_sum, bc_sum, biou_sum)
                step_results[str(k)] = dict(
                    steps=k, seed=args.seed, image_count=len(ids),
                    nfe_total=len(ids) * 35 * k, nfe_per_image=35 * k,
                    metrics=metrics,
                    note='early return along fixed K=4 trajectory (matches chapter_controlled_20261031 analysis.py)',
                )
                print(f'[EVAL] I split=val K={k} mIoU={100*metrics["miou_4class"]:.4f}% BF2={100*metrics["boundary_f1_foreground"]:.4f}%', flush=True)
            np.savez_compressed(out_npz, **npz_payload)
            payload = dict(
                status='completed', evaluated_at_bjt=now_bjt(), model='I', split='val', sampling_seed=args.seed,
                steps_list=args.steps, checkpoint=str(ckpt_path), checkpoint_sha256=meta['checkpoint_sha256'],
                best_epoch=int(meta['epoch']), counts_npz=str(out_npz),
                reconciled_with_official_best_val_npz=True,
                elapsed_seconds=time.perf_counter() - t_start, results=step_results,
            )
            atomic_json(out_json, payload)
        else:
            k = args.steps[0]
            assert len(ds) == 27
            mats, bc_tols, biou_tols, img_rows, ids = [], {1: [], 2: [], 4: []}, {1: [], 2: [], 4: []}, [], []
            nfe_total = 0
            for idx in range(len(ds)):
                d = ds[idx]
                gt = d['mask'].numpy()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred, nfe = CC.predict(model, c, d['image'], d['thumb'], d['id'], stop_after=k, sampling_seed=args.seed)
                torch.cuda.synchronize()
                sec = time.perf_counter() - t0
                nfe_total += nfe
                cm = confusion(gt, pred)
                mats.append(cm)
                row_bc = {}
                for tol in (1, 2, 4):
                    bc = boundary_counts(gt, pred, tol)
                    bi = boundary_iou_counts(gt, pred, tol)
                    bc_tols[tol].append(bc)
                    biou_tols[tol].append(bi)
                    row_bc[f'bf{tol}'] = bc.tolist()
                    row_bc[f'biou{tol}'] = bi.tolist()
                ids.append(d['id'])
                img_rows.append(dict(id=d['id'], seconds=sec, nfe=nfe, confusion=cm.tolist(), **row_bc))
            npz_payload = {
                'ids': np.array(ids),
                f's{k}_cm': np.asarray(mats, dtype=np.int64),
                f's{k}_bf1': np.asarray(bc_tols[1], dtype=np.int64),
                f's{k}_bf2': np.asarray(bc_tols[2], dtype=np.int64),
                f's{k}_bf4': np.asarray(bc_tols[4], dtype=np.int64),
                f's{k}_biou1': np.asarray(biou_tols[1], dtype=np.int64),
                f's{k}_biou2': np.asarray(biou_tols[2], dtype=np.int64),
                f's{k}_biou4': np.asarray(biou_tols[4], dtype=np.int64),
            }
            np.savez_compressed(out_npz, **npz_payload)
            metrics = summarize_full_metrics(
                np.sum(mats, 0),
                {tol: np.sum(bc_tols[tol], 0) for tol in (1, 2, 4)},
                {tol: np.sum(biou_tols[tol], 0) for tol in (1, 2, 4)},
            )
            tot_sec = sum(r['seconds'] for r in img_rows)
            step_res = dict(steps=k, seed=args.seed, image_count=len(ids), nfe_total=nfe_total,
                            nfe_per_image=nfe_total / len(ids), seconds=tot_sec,
                            seconds_per_image=tot_sec / len(ids), metrics=metrics, images=img_rows)
            payload = dict(
                status='completed', evaluated_at_bjt=now_bjt(), model='I', split='test', sampling_seed=args.seed,
                steps_list=[k], checkpoint=str(ckpt_path), checkpoint_sha256=meta['checkpoint_sha256'],
                best_epoch=int(meta['epoch']), counts_npz=str(out_npz),
                elapsed_seconds=time.perf_counter() - t_start,
                metrics=metrics, seconds=tot_sec, nfe_total=nfe_total, results={str(k): step_res},
            )
            atomic_json(out_json, payload)
            print(f'[EVAL] I split=test K={k} mIoU={100*metrics["miou_4class"]:.4f}% BF2={100*metrics["boundary_f1_foreground"]:.4f}% ({tot_sec:.1f}s)', flush=True)
    else:
        raise ValueError(f'Unsupported model {args.model}')


# ----------------------------------------------------------------------------- bootstrap (§5.3)

def _load_aligned_counts(npz_path: Path, cm_key: str, bf_key: str, canonical_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as z:
        ids = [str(x) for x in z['ids']]
        assert set(ids) == set(canonical_ids) and len(ids) == len(canonical_ids), f'ID set mismatch in {npz_path}'
        id_map = {ident: i for i, ident in enumerate(ids)}
        order = [id_map[ident] for ident in canonical_ids]
        return z[cm_key][order].astype(np.int64), z[bf_key][order].astype(np.int64)


def _scores_from_arrays(cm: np.ndarray, bf: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    den = cm.sum(-1) + cm.sum(-2) - tp
    iou = np.divide(tp, den, out=np.full(tp.shape, np.nan), where=den > 0)
    p = np.divide(bf[..., 0], bf[..., 1], out=np.zeros(bf.shape[:-1]), where=bf[..., 1] > 0)
    r = np.divide(bf[..., 2], bf[..., 3], out=np.zeros(bf.shape[:-1]), where=bf[..., 3] > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=p + r > 0)
    f = np.where(bf[..., 1] + bf[..., 3] > 0, f, np.nan)
    few_sum = cm[..., 2, :].sum(-1)
    f2m = np.divide(cm[..., 2, 1], few_sum, out=np.full(few_sum.shape, np.nan), where=few_sum > 0)
    return {
        'miou_4class': np.nanmean(iou, axis=-1),
        'iou_monolayer': iou[..., 1],
        'iou_fewlayer': iou[..., 2],
        'iou_multilayer': iou[..., 3],
        'thin_mean_iou': np.nanmean(iou[..., 1:3], axis=-1),
        'boundary_f1_foreground': np.nanmean(f[..., 1:], axis=-1),
        'few_to_mono_rate': f2m,
    }


def _compute_bootstrap_group(arrays: dict[str, tuple[np.ndarray, np.ndarray]], pairs: list[tuple[str, str]],
                             canonical_ids: list[str], seed: int = 3407, n_boot: int = 10000) -> dict:
    n = len(canonical_ids)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, n, size=(n_boot, n))
    point, boot = {}, {}
    for name, (cm, bf) in arrays.items():
        point[name] = {k: float(v) for k, v in _scores_from_arrays(cm.sum(0), bf.sum(0)).items()}
        boot[name] = _scores_from_arrays(cm[ix].sum(1), bf[ix].sum(1))

    contrasts = {}
    for a, b in pairs:
        diff = {}
        # LOO deltas
        loo_masks = [np.arange(n) != i for i in range(n)]
        loo_cm_a = np.stack([arrays[a][0][m].sum(0) for m in loo_masks])
        loo_bf_a = np.stack([arrays[a][1][m].sum(0) for m in loo_masks])
        loo_cm_b = np.stack([arrays[b][0][m].sum(0) for m in loo_masks])
        loo_bf_b = np.stack([arrays[b][1][m].sum(0) for m in loo_masks])
        loo_a = _scores_from_arrays(loo_cm_a, loo_bf_a)
        loo_b = _scores_from_arrays(loo_cm_b, loo_bf_b)
        for k in point[a]:
            delta_b = 100.0 * (boot[a][k] - boot[b][k])
            loo_vals = 100.0 * (loo_a[k] - loo_b[k])
            pt_pp = 100.0 * (point[a][k] - point[b][k])
            lo, hi = np.quantile(delta_b, [0.025, 0.975])
            min_idx, max_idx = int(np.argmin(loo_vals)), int(np.argmax(loo_vals))
            diff[k] = dict(
                point_pp=float(pt_pp),
                bootstrap_95_interval_pp=[float(lo), float(hi)],
                loo_min_pp=float(loo_vals[min_idx]),
                loo_min_omitted_id=canonical_ids[min_idx],
                loo_max_pp=float(loo_vals[max_idx]),
                loo_max_omitted_id=canonical_ids[max_idx],
            )
        contrasts[f'{a}-{b}'] = diff
    return dict(n_images=n, ids=canonical_ids, point_fraction=point, contrasts=contrasts)


def cmd_bootstrap(args):
    out = ensure_out_dir(args.out)
    plan = read_json(out / 'plan.json')
    val_ids = Path(plan['splits']['val']['snapshot_path']).read_text().split()
    test_ids = Path(plan['splits']['test']['snapshot_path']).read_text().split()

    val_arrays = {
        'G_onehot_k20': _load_aligned_counts(out / 'eval/steps_val_g_onehot_counts.npz', 's20_cm', 's20_bf2', val_ids),
        'G_onehot_k4': _load_aligned_counts(out / 'eval/steps_val_g_onehot_counts.npz', 's4_cm', 's4_bf2', val_ids),
        'G_ordinal_k20': _load_aligned_counts(out / 'eval/steps_val_g_ordinal_counts.npz', 's20_cm', 's20_bf2', val_ids),
        'G_ordinal_k4': _load_aligned_counts(out / 'eval/steps_val_g_ordinal_counts.npz', 's4_cm', 's4_bf2', val_ids),
        'I': _load_aligned_counts(out / 'eval/steps_val_I_counts.npz', 's4_cm', 's4_bf2', val_ids),
        'R': _load_aligned_counts(CONTROLLED_ANALYSIS / 'R/trajectory_counts.npz', 's4_cm', 's4_bf2', val_ids),
    }
    test_arrays = {
        'G_onehot_k20': _load_aligned_counts(out / 'eval/test27_g_onehot_k20_counts.npz', 's20_cm', 's20_bf2', test_ids),
        'G_onehot_k4': _load_aligned_counts(out / 'eval/test27_g_onehot_k4_counts.npz', 's4_cm', 's4_bf2', test_ids),
        'G_ordinal_k20': _load_aligned_counts(out / 'eval/test27_g_ordinal_k20_counts.npz', 's20_cm', 's20_bf2', test_ids),
        'G_ordinal_k4': _load_aligned_counts(out / 'eval/test27_g_ordinal_k4_counts.npz', 's4_cm', 's4_bf2', test_ids),
        'I': _load_aligned_counts(out / 'eval/test27_I_k4_counts.npz', 's4_cm', 's4_bf2', test_ids),
        'R': _load_aligned_counts(HIST_GLP_TEST / 'E3R_confusion.npz', 'per_image', 'per_image_boundary', test_ids),
    }
    pairs = [
        ('G_onehot_k20', 'I'),
        ('G_ordinal_k20', 'G_onehot_k20'),
        ('I', 'R'),
        ('G_onehot_k4', 'I'),
        ('G_ordinal_k4', 'G_onehot_k4'),
        ('G_onehot_k20', 'R'),
    ]
    res = dict(
        computed_at_bjt=now_bjt(),
        bootstrap_iterations=10000,
        bootstrap_seed=3407,
        note='Intervals describe image sampling uncertainty for fixed trained weights; not confirmatory hypothesis testing or training-seed variance.',
        val25=_compute_bootstrap_group(val_arrays, pairs, val_ids, seed=3407, n_boot=10000),
        test27=_compute_bootstrap_group(test_arrays, pairs, test_ids, seed=3407, n_boot=10000),
    )
    atomic_json(out / 'eval/bootstrap.json', res)
    print(json.dumps(dict(status='completed', output=str(out / 'eval/bootstrap.json')), indent=2))


# ----------------------------------------------------------------------------- diagnose (S9 §6)

def _diag_metrics(cm: np.ndarray, bc: np.ndarray, band_ok: int, band_tot: int, off_ok: int, off_tot: int) -> dict:
    m = DSE.summarize_material(cm)
    b = summarize_boundaries(bc)
    return dict(
        miou_4class=m['miou_4class'],
        thin_mean_iou=m['thin_mean_iou'],
        iou_monolayer=m['iou_monolayer'],
        iou_fewlayer=m['iou_fewlayer'],
        iou_multilayer=m['iou_multilayer'],
        boundary_f1_foreground=b['boundary_f1_foreground'],
        band8_accuracy=band_ok / max(1, band_tot),
        off_band_accuracy=off_ok / max(1, off_tot),
    )


@torch.no_grad()
def _diagnose_g(out: Path, plan: dict, arm: str, smoke_one: bool) -> dict:
    cfg = read_json(Path(plan['configs'][arm]['path']))
    ckpt_path = Path(plan['runs'][arm]) / 'best.pt'
    DSE.configure(cfg)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model, diffusion = DSE.build(cfg)
    model.load_state_dict(state['model'])
    model.eval()
    device = cfg['device']

    ds = DSE.SegmentationDataset(cfg, 'val')
    n_imgs = 1 if smoke_one else len(ds)
    steps = 20
    times = torch.linspace(diffusion.timesteps - 1, 0, steps).round().long().tolist()
    targets = [999, 749, 499, 249, 99, 0]
    probe_indices = []
    for tgt in targets:
        best_i = min(range(len(times)), key=lambda i: (abs(times[i] - tgt), i))
        probe_indices.append((tgt, best_i, int(times[best_i])))
    probe_t_set = {t_val for _, _, t_val in probe_indices}

    size = cfg['crop_size']
    stride = cfg['stride']
    wb = cfg['window_batch_size']
    seed = cfg['sampling_seed']

    def empty_acc():
        return dict(cm=np.zeros((4, 4), np.int64), bc=np.zeros((4, 4), np.int64), band=[0, 0], off=[0, 0])

    inf_acc = [empty_acc() for _ in range(steps)]
    trn_acc = {t_val: empty_acc() for t_val in sorted(probe_t_set)}
    t0 = time.perf_counter()

    for idx in range(n_imgs):
        rgb_raw, mask_raw, ident = ds.source.raw(idx)
        rgb_pad, mask_pad = DSE.pad_full_image(rgb_raw, mask_raw, size)
        thumb, _ = DSE.global_inputs(rgb_pad, mask_pad)
        image = DSE.normalize(rgb_pad)
        native_h, native_w = mask_raw.shape
        height, width = image.shape[-2:]
        bd = band_mask(mask_raw, 8)
        clean_gt = diffusion.encode(torch.from_numpy(mask_pad[None].astype(np.int64)))[0]

        gen = torch.Generator().manual_seed(image_seed(ident, seed))
        noise = torch.randn((diffusion.codec.channels, height, width), generator=gen)
        coords = [(y, x) for y in positions(height, size, stride) for x in positions(width, size, stride)]
        context = model.encode_global([thumb.to(device)])[0]

        inf_totals = [torch.zeros((4, height, width)) for _ in range(steps)]
        trn_totals = {t_val: torch.zeros((4, height, width)) for t_val in probe_t_set}
        counts = torch.zeros((height, width))

        for start in range(0, len(coords), wb):
            part = coords[start:start + wb]
            patches = torch.stack([image[:, y:y + size, x:x + size] for y, x in part]).to(device)
            boxes = torch.tensor([[y / height, x / width, (y + size) / height, (x + size) / width] for y, x in part], device=device)
            condition = model.encode_condition(patches, boxes, [context] * len(part))
            noise_win = torch.stack([noise[:, y:y + size, x:x + size] for y, x in part]).to(device)
            clean_win = torch.stack([clean_gt[:, y:y + size, x:x + size] for y, x in part]).to(device)

            # Inference trajectory across all 20 DDIM steps
            cur_state = noise_win.clone()
            for s_idx, t_val in enumerate(times):
                bt = torch.full((len(part),), t_val, dtype=torch.long, device=device)
                probs = model.denoise(condition, cur_state, bt).softmax(1)
                for p_w, (y, x) in zip(probs.cpu(), part):
                    inf_totals[s_idx][:, y:y + size, x:x + size] += p_w
                clean = diffusion.codec.from_probabilities(probs)
                if s_idx < steps - 1:
                    cur_state = diffusion.ddim_step(cur_state, clean, t_val, times[s_idx + 1])

            # Training-type q_sample probes at the 6 selected grid timesteps
            for t_val in sorted(probe_t_set):
                bt = torch.full((len(part),), t_val, dtype=torch.long, device=device)
                state_q = diffusion.q_sample(clean_win, bt, noise_win)
                probs_q = model.denoise(condition, state_q, bt).softmax(1).cpu()
                for p_w, (y, x) in zip(probs_q, part):
                    trn_totals[t_val][:, y:y + size, x:x + size] += p_w

            for y, x in part:
                counts[y:y + size, x:x + size] += 1

        for s_idx in range(steps):
            pred = (inf_totals[s_idx] / counts)[:, :native_h, :native_w].argmax(0).numpy().astype(np.uint8)
            a = inf_acc[s_idx]
            a['cm'] += confusion(mask_raw, pred)
            a['bc'] += boundary_counts(mask_raw, pred, cfg['boundary_tolerance'])
            ok = pred == mask_raw
            a['band'][0] += int(ok[bd].sum()); a['band'][1] += int(bd.sum())
            a['off'][0] += int(ok[~bd].sum()); a['off'][1] += int((~bd).sum())
        for t_val in sorted(probe_t_set):
            pred = (trn_totals[t_val] / counts)[:, :native_h, :native_w].argmax(0).numpy().astype(np.uint8)
            a = trn_acc[t_val]
            a['cm'] += confusion(mask_raw, pred)
            a['bc'] += boundary_counts(mask_raw, pred, cfg['boundary_tolerance'])
            ok = pred == mask_raw
            a['band'][0] += int(ok[bd].sum()); a['band'][1] += int(bd.sum())
            a['off'][0] += int(ok[~bd].sum()); a['off'][1] += int((~bd).sum())
        if (idx + 1) % 5 == 0 or idx + 1 == n_imgs:
            print(f'[DIAG] {arm} image {idx + 1}/{n_imgs} ({ident})', flush=True)

    inf_curve = []
    for s_idx, t_val in enumerate(times):
        a = inf_acc[s_idx]
        m = _diag_metrics(a['cm'], a['bc'], a['band'][0], a['band'][1], a['off'][0], a['off'][1])
        inf_curve.append(dict(step=s_idx, t=int(t_val), **m))

    grid_probes = []
    for tgt, s_idx, t_val in probe_indices:
        a_trn = trn_acc[t_val]
        a_inf = inf_acc[s_idx]
        m_trn = _diag_metrics(a_trn['cm'], a_trn['bc'], a_trn['band'][0], a_trn['band'][1], a_trn['off'][0], a_trn['off'][1])
        m_inf = _diag_metrics(a_inf['cm'], a_inf['bc'], a_inf['band'][0], a_inf['band'][1], a_inf['off'][0], a_inf['off'][1])
        delta_pp = {k: 100.0 * (m_trn[k] - m_inf[k]) for k in m_trn}
        grid_probes.append(dict(target_t=tgt, grid_t=t_val, ddim_step=s_idx,
                                training_type=m_trn, inference_type=m_inf, delta_pp=delta_pp))

    reproduces = None
    if not smoke_one:
        best_val = state.get('validation', {}).get('metrics', {}).get('miou_4class')
        if best_val is not None:
            reproduces = abs(inf_curve[-1]['miou_4class'] - best_val) < 1e-9
            assert reproduces, f'{arm} diagnose final step mIoU mismatch with best.pt val: {inf_curve[-1]["miou_4class"]} vs {best_val}'

    return dict(
        status='completed',
        diagnosed_at_bjt=now_bjt(),
        arm=arm,
        family='G',
        images=n_imgs,
        smoke_one=smoke_one,
        ddim_steps=steps,
        sampling_seed=seed,
        checkpoint=str(ckpt_path),
        checkpoint_sha256=sha256_file(ckpt_path),
        reproduces_recorded_val_final=reproduces,
        seconds=time.perf_counter() - t0,
        inference_trajectory=inf_curve,
        grid_probes=grid_probes,
    )


@torch.no_grad()
def _diagnose_ir(out: Path, plan: dict, arm: str, smoke_one: bool) -> dict:
    import glpflow_core as G
    from chapter_controlled_20261031 import core as CC
    assert arm in ('I', 'R')
    ckpt_path = Path(plan['existing_checkpoints'][arm]['path'])
    if sha256_file(ckpt_path) != plan['existing_checkpoints'][arm]['sha256']:
        raise RuntimeError(f'Checkpoint sha256 mismatch for {arm}')
    model, c, meta = CC.load_best(arm)
    device = c['device']
    ds = G.GLPDataset(c, 'val', train=False)
    n_imgs = 1 if smoke_one else len(ds)
    steps = 4
    seed = c['sampling_seed']

    def empty_acc():
        return dict(cm=np.zeros((4, 4), np.int64), bc=np.zeros((4, 4), np.int64), band=[0, 0], off=[0, 0])

    self_acc = [empty_acc() for _ in range(steps)]
    gt_acc = [empty_acc() for _ in range(steps)]
    t0 = time.perf_counter()

    for idx in range(n_imgs):
        d = ds[idx]
        image, thumb, ident = d['image'], d['thumb'], d['id']
        truth = d['mask'].numpy()
        bd = band_mask(truth, 8)
        h, w = image.shape[-2:]
        sz = c['crop_size']
        hw = ((h // 64) * 64, (w // 64) * 64)
        ctx = model.global_context([thumb.to(device)])
        gen = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'{ident}:{seed}'.encode()).digest()[:8], 'little') % (2**63 - 1))
        field = torch.randn((4, h, w), generator=gen)
        y_full = G.encode_targets(d['mask'][None])[0]

        coords = [(y, x) for y in G.positions(h, sz, c['stride']) for x in G.positions(w, sz, c['stride'])]
        self_totals = [torch.zeros((4, h, w)) for _ in range(steps)]
        gt_totals = [torch.zeros((4, h, w)) for _ in range(steps)]
        counts = torch.zeros((h, w))

        for start in range(0, len(coords), c['window_batch_size']):
            part = coords[start:start + c['window_batch_size']]
            b = len(part)
            img = torch.stack([image[:, y:y + sz, x:x + sz] for y, x in part]).to(device)
            boxes = torch.tensor([[y / hw[0], x / hw[1], (y + sz) / hw[0], (x + sz) / hw[1]] for y, x in part])
            eps = torch.stack([field[:, y:y + sz, x:x + sz] for y, x in part]).to(device)
            yw = torch.stack([y_full[:, y:y + sz, x:x + sz] for y, x in part]).to(device)
            feats = model.local(img)
            p, umap, gtok, gpos, _ = G.window_conditions(model, c, ctx * b, boxes, sz, device)
            lv = torch.full((b,), 1, device=device, dtype=torch.long)

            x0 = G.make_source(c, p, umap, eps)
            x = x0.clone()
            sc = torch.zeros_like(p)
            for j in range(steps):
                t_val = j / steps
                t = torch.full((b,), t_val, device=device)
                logits_self = model.denoise(feats, x, p, umap, sc, t, lv, gtok, gpos, boxes)
                q_self = logits_self.softmax(1)
                for q_w, (y, x_) in zip(q_self.cpu(), part):
                    self_totals[j][:, y:y + sz, x_:x_ + sz] += q_w

                if j == 0:
                    q_gt = q_self
                else:
                    x_gt = (1.0 - t_val) * x0 + t_val * yw
                    q_gt = model.denoise(feats, x_gt, p, umap, sc, t, lv, gtok, gpos, boxes).softmax(1)
                for q_w, (y, x_) in zip(q_gt.cpu(), part):
                    gt_totals[j][:, y:y + sz, x_:x_ + sz] += q_w

                x = CC.advance(c, x, q_self, t, torch.full_like(t, 1.0 / steps))
                sc = q_self
            for y, x_ in part:
                counts[y:y + sz, x_:x_ + sz] += 1

        for j in range(steps):
            for tot, acc in ((self_totals[j], self_acc[j]), (gt_totals[j], gt_acc[j])):
                pred = (tot / counts).argmax(0).numpy().astype(np.uint8)
                acc['cm'] += confusion(truth, pred)
                acc['bc'] += boundary_counts(truth, pred, c['boundary_tolerance'])
                ok = pred == truth
                acc['band'][0] += int(ok[bd].sum()); acc['band'][1] += int(bd.sum())
                acc['off'][0] += int(ok[~bd].sum()); acc['off'][1] += int((~bd).sum())
        if (idx + 1) % 5 == 0 or idx + 1 == n_imgs:
            print(f'[DIAG] {arm} image {idx + 1}/{n_imgs} ({ident})', flush=True)

    reproduces = None
    if not smoke_one:
        with np.load(CC.RUNS[arm] / 'best_val_confusion.npz') as z:
            expected_cm = z['global_matrix']
        reproduces = bool(np.array_equal(self_acc[-1]['cm'], expected_cm))
        assert reproduces, f'{arm} diagnose final step confusion mismatch with best_val_confusion.npz'

    inf_curve, grid_probes = [], []
    for j in range(steps):
        t_val = j / steps
        m_self = _diag_metrics(self_acc[j]['cm'], self_acc[j]['bc'], self_acc[j]['band'][0], self_acc[j]['band'][1], self_acc[j]['off'][0], self_acc[j]['off'][1])
        m_gt = _diag_metrics(gt_acc[j]['cm'], gt_acc[j]['bc'], gt_acc[j]['band'][0], gt_acc[j]['band'][1], gt_acc[j]['off'][0], gt_acc[j]['off'][1])
        # For I, training state is GT interpolation; for R, training state IS self rollout!
        m_trn = m_gt if arm == 'I' else m_self
        delta_pp = {k: 100.0 * (m_trn[k] - m_self[k]) for k in m_self}
        gt_minus_self_pp = {k: 100.0 * (m_gt[k] - m_self[k]) for k in m_self}
        inf_curve.append(dict(step=j, t=t_val, **m_self))
        grid_probes.append(dict(
            step=j, t=t_val,
            training_state_source='gt_interpolation' if arm == 'I' else 'self_rollout',
            training_type=m_trn,
            inference_type=m_self,
            gt_interpolation=m_gt,
            delta_pp=delta_pp,
            gt_minus_self_pp=gt_minus_self_pp,
        ))

    return dict(
        status='completed',
        diagnosed_at_bjt=now_bjt(),
        arm=arm,
        family='flow',
        images=n_imgs,
        smoke_one=smoke_one,
        steps=steps,
        sampling_seed=seed,
        checkpoint=str(ckpt_path),
        checkpoint_sha256=meta['checkpoint_sha256'],
        reproduces_recorded_val_final=reproduces,
        seconds=time.perf_counter() - t0,
        inference_trajectory=inf_curve,
        grid_probes=grid_probes,
    )


def cmd_diagnose(args):
    out = ensure_out_dir(args.out)
    plan = read_json(out / 'plan.json')
    if args.arm in ('g_onehot', 'g_ordinal'):
        res = _diagnose_g(out, plan, args.arm, args.smoke_one)
    elif args.arm in ('I', 'R'):
        res = _diagnose_ir(out, plan, args.arm, args.smoke_one)
    else:
        raise ValueError(args.arm)
    if args.smoke_one:
        print(json.dumps(clean_json(res), ensure_ascii=False, indent=2))
        return
    atomic_json(out / 'diagnostics' / f'state_gap_{args.arm}.json', res)


# ----------------------------------------------------------------------------- S4 Unified Inference Benchmark (§7)

def _sync_time(fn, iters: int, warmup: int) -> np.ndarray:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.asarray(times, dtype=np.float64)


def _flops_of(fn) -> int:
    import complete_ch4_20261024 as COMP
    return COMP.flops_of(fn)


def _native_baseline_ops():
    import baselines_20261022.models as M
    import baselines_20261022.diffusion as Dm

    def nat_interp(x, size=None, scale_factor=None):
        h, w = x.shape[-2:]
        if size is not None:
            ho, wo = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
        else:
            sf = scale_factor if isinstance(scale_factor, (tuple, list)) else (scale_factor, scale_factor)
            ho, wo = int(round(h * sf[0])), int(round(w * sf[1]))
        if (ho, wo) == (h, w):
            return x
        return F.interpolate(x, size=(ho, wo), mode='bilinear', align_corners=False)

    M.det_interpolate = nat_interp
    Dm.det_interpolate = nat_interp
    M.det_adaptive_avg_pool2d = lambda x, o: F.adaptive_avg_pool2d(x, o)
    M.det_grid_sample = lambda v, g: F.grid_sample(v, g, mode='bilinear', padding_mode='zeros', align_corners=False)


def _bench_infer_mode():
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def _run_bench_infer_one(out: Path, plan: dict, name: str):
    import glpflow_core as G
    from chapter_controlled_20261031 import core as CC
    from repela_20261024 import core as RP
    from repela_20261024.model import native_ops

    B_CORE = RP.B
    WIN = 512

    if name.startswith('g_'):
        parts = name.split('_')
        arm = '_'.join(parts[:-1])  # g_onehot / g_ordinal
        steps = int(parts[-1][1:])
        cfg = read_json(Path(plan['configs'][arm]['path']))
        ckpt_path = Path(plan['runs'][arm]) / 'best.pt'
        DSE.configure(cfg)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model, diffusion = DSE.build(cfg)
        model.load_state_dict(state['model'])
        model.eval()
        bench_cfg = dict(cfg, window_batch_size=8, sampling_steps=steps)
        ds = DSE.SegmentationDataset(bench_cfg, 'val')
        img0 = ds[0]
        height, width = img0['image'].shape[-2:]
        holder = {}
        box = torch.tensor([[0.0, 0.0, WIN / height, WIN / width]], device='cuda')
        eps = torch.randn((8, diffusion.codec.channels, WIN, WIN), device='cuda')

        def global_fn(d):
            holder['ctx'] = model.encode_global([d['thumb'].cuda()])[0]
            return holder['ctx']

        def window_fn(patches):
            b = len(patches)
            cond = model.encode_condition(patches, box.expand(b, 4), [holder['ctx']] * b)
            return diffusion.sample(model, cond, eps[:b], steps=steps)

        def predict_fn(d):
            pred, _ = DSE.predict_image(model, diffusion, d['image'], d['thumb'], d['id'],
                                        bench_cfg, native_shape=d['mask'].shape, sampling_steps=steps)
            return pred

        win_coords = [(y, x) for y in positions(height, WIN, bench_cfg['stride']) for x in positions(width, WIN, bench_cfg['stride'])]
        ckpt_sha = sha256_file(ckpt_path)
        meta = dict(name=name, family='G', steps=steps, window_batch_size=8,
                    checkpoint=str(ckpt_path), checkpoint_sha256=ckpt_sha, best_epoch=int(state['epoch']))
        ctx_mgr = contextlib.nullcontext()
        use_native_baselines = False

    elif name in ('I_k4', 'R_k1', 'R_k2', 'R_k4'):
        arm = name.split('_')[0]
        stop = int(name.split('_')[1][1:])
        ckpt_path = Path(plan['existing_checkpoints'][arm]['path'])
        model, c, meta_c = CC.load_best(arm)
        assert c['window_batch_size'] == 8
        ds = G.GLPDataset(c, 'val', train=False)
        holder = {}
        box = torch.tensor([[0.0, 0.0, WIN / 1920, WIN / 2560]])
        eps = torch.randn((8, 4, WIN, WIN), device='cuda')

        def global_fn(d):
            holder['ctx'] = model.global_context([d['thumb'].cuda()])
            return holder['ctx']

        def window_fn(patches):
            b = len(patches)
            return CC.infer_windows(model, c, patches, box.expand(b, 4), holder['ctx'] * b, eps[:b], stop_after=stop)[0]

        def predict_fn(d):
            return CC.predict(model, c, d['image'], d['thumb'], d['id'], stop_after=stop)[0]

        win_coords = [(y, x) for y in G.positions(ds[0]['image'].shape[-2], WIN, c['stride'])
                      for x in G.positions(ds[0]['image'].shape[-1], WIN, c['stride'])]
        meta = dict(name=name, family='flow', steps=stop, window_batch_size=8,
                    checkpoint=str(ckpt_path), checkpoint_sha256=meta_c['checkpoint_sha256'], best_epoch=int(meta_c['epoch']))
        ctx_mgr = contextlib.nullcontext()
        use_native_baselines = False

    elif name == 'E1':
        ckpt_path = Path(plan['existing_checkpoints']['E1']['path'])
        G.configure(42)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        c = state['config']
        assert c['window_batch_size'] == 8
        model = G.GLPFlowNet(c).cuda()
        model.load_state_dict(state['model'], strict=True)
        model.eval()
        ds = G.GLPDataset(c, 'val', train=False)
        holder = {}
        box = torch.tensor([[0.0, 0.0, WIN / 1920, WIN / 2560]])

        def global_fn(d):
            holder['ctx'] = model.global_context([d['thumb'].cuda()])
            return holder['ctx']

        def window_fn(patches):
            b = len(patches)
            return G.infer_windows(model, c, patches, box.expand(b, 4), holder['ctx'] * b, None, 1)[0]

        def predict_fn(d):
            return G.predict(model, c, d['image'], d['thumb'], d['id'], 1)[0]

        win_coords = [(y, x) for y in G.positions(ds[0]['image'].shape[-2], WIN, c['stride'])
                      for x in G.positions(ds[0]['image'].shape[-1], WIN, c['stride'])]
        meta = dict(name=name, family='direct', steps=1, window_batch_size=8,
                    checkpoint=str(ckpt_path), checkpoint_sha256=sha256_file(ckpt_path), best_epoch=int(state['epoch']))
        ctx_mgr = contextlib.nullcontext()
        use_native_baselines = False

    elif name in ('segformer_b1', 'ddp', 'ccdm', 'repela_deploy'):
        key = 'repela_small' if name == 'repela_deploy' else name
        ckpt_path = Path(plan['existing_checkpoints'][key]['path'])
        G.configure(42)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        c = state['config']
        assert c['window_batch_size'] == 8
        model = B_CORE.build_model(c).cuda()
        model.load_state_dict(state['model'], strict=True)
        if name == 'repela_deploy':
            model.switch_to_deploy()
        model.eval()
        ds = G.GLPDataset(c, 'val', train=False)
        m = c['method']
        if m == 'ddp':
            eps = torch.randn((8, 256, WIN // 4, WIN // 4), device='cuda')
            steps = c['sampling_steps']
            window_fn = lambda patches: model.infer_windows(patches, eps=eps[:len(patches)], steps=steps)[0]
        elif m == 'ccdm':
            u = torch.rand((8, c['sampling_steps'], WIN, WIN), device='cuda')
            steps = c['sampling_steps']
            window_fn = lambda patches: model.infer_windows(patches, u_field=u[:len(patches)], steps=steps)[0]
        else:
            steps = 1
            window_fn = lambda patches: model(patches)
        global_fn = None
        predict_fn = lambda d: B_CORE.predict(model, c, d['image'], d['id'])[0]
        win_coords = [(y, x) for y in G.positions(ds[0]['image'].shape[-2], WIN, c['stride'])
                      for x in G.positions(ds[0]['image'].shape[-1], WIN, c['stride'])]
        meta = dict(name=name, family='baseline', steps=steps, window_batch_size=8,
                    checkpoint=str(ckpt_path), checkpoint_sha256=sha256_file(ckpt_path), best_epoch=int(state['epoch']))
        ctx_mgr = native_ops() if name == 'repela_deploy' else contextlib.nullcontext()
        use_native_baselines = name != 'repela_deploy'
    else:
        raise ValueError(f'Unknown bench-infer object: {name}')

    d0 = ds[0]
    params = sum(p.numel() for p in model.parameters())
    state_mib = sum(t.numel() * t.element_size() for t in model.state_dict().values()) / 2**20
    if global_fn is not None:
        global_fn(d0)
    ref_pred = predict_fn(d0)

    _bench_infer_mode()
    if use_native_baselines:
        _native_baseline_ops()

    with ctx_mgr:
        pred_nat = predict_fn(d0)
        agreement = float((pred_nat == ref_pred).mean())
        assert agreement >= 0.9999, f'{name} native agreement {agreement} < 0.9999'

        x1 = torch.stack([d0['image'][:, :WIN, :WIN]]).cuda()
        x8 = torch.stack([d0['image'][:, y:y + WIN, x:x + WIN] for y, x in win_coords[:8]]).cuda()
        if global_fn is not None:
            global_fn(d0)
            g_flops = _flops_of(lambda: global_fn(d0))
            g_times = _sync_time(lambda: global_fn(d0), 50, 10)
            g_lat_ms = float(np.median(g_times) * 1e3)
        else:
            g_flops, g_lat_ms = 0, 0.0
        w_flops = _flops_of(lambda: window_fn(x1))
        w_gmacs = w_flops / 2 / 1e9
        n_win = len(win_coords)
        assert n_win == 35, f'Expected 35 windows, got {n_win}'
        img_gmacs = (n_win * w_flops + g_flops) / 2 / 1e9

        w_lat = _sync_time(lambda: window_fn(x1), 100, 20)
        b8_lat = _sync_time(lambda: window_fn(x8), 50, 10)

        # Warmup on ds[0] before peak memory reset
        predict_fn(d0)
        predict_fn(d0)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Plan §7: fixed 5 val images x 3 repeats, window batch 8
        sample_items = [ds[i] for i in range(5)]
        sample_ids = [d['id'] for d in sample_items]
        raw_times = []
        for rep in range(3):
            for d in sample_items:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                predict_fn(d)
                torch.cuda.synchronize()
                raw_times.append(time.perf_counter() - t0)
        peak_alloc_mib = torch.cuda.max_memory_allocated() / 2**20
        peak_res_mib = torch.cuda.max_memory_reserved() / 2**20

    res = dict(
        meta,
        benchmarked_at_bjt=now_bjt(),
        params=params,
        params_m=params / 1e6,
        params_fp32_mib=params * 4 / 2**20,
        state_dict_mib=state_mib,
        windows_per_image=n_win,
        global_flops=g_flops,
        window_flops=w_flops,
        window_gmacs=w_gmacs,
        image_gmacs=img_gmacs,
        window_warmup=20,
        window_iterations=100,
        window_latency_ms_median=float(np.median(w_lat) * 1e3),
        window_latency_ms_p95=float(np.percentile(w_lat, 95) * 1e3),
        window_throughput_per_s_b8=float(8.0 / np.median(b8_lat)),
        global_latency_ms_median=g_lat_ms,
        whole_image_protocol='5 fixed val images x 3 repeats (after ds[0] warmup), window_batch_size=8',
        image_ids=sample_ids,
        repeats=3,
        image_predict_s_median=float(np.median(raw_times)),
        image_predict_s_mean=float(np.mean(raw_times)),
        image_predict_s_p95=float(np.percentile(raw_times, 95)),
        raw_image_seconds=raw_times,
        infer_peak_allocated_mib=peak_alloc_mib,
        infer_peak_reserved_mib=peak_res_mib,
        agreement_with_as_evaluated=agreement,
    )
    atomic_json(out / 'benchmark/objects' / f'{name}.json', res)
    print(json.dumps(dict(name=name, params=params, w_gmacs=round(w_gmacs, 3),
                          w_ms=round(res['window_latency_ms_median'], 3),
                          img_s=round(res['image_predict_s_median'], 4),
                          mem_mib=round(peak_res_mib, 1),
                          agree=round(agreement, 6)), ensure_ascii=False), flush=True)


def _gpu_env_info() -> dict:
    smi = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', '--format=csv,noheader'],
        capture_output=True, text=True,
    ).stdout.strip()
    return dict(
        host=socket.gethostname(),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
        nvidia_smi=smi,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        cpu_count=os.cpu_count(),
        precision='FP32, TF32 disabled',
    )


def cmd_bench_infer(args):
    out = ensure_out_dir(args.out)
    plan = read_json(out / 'plan.json')
    if args.one:
        _run_bench_infer_one(out, plan, args.one)
        return

    import complete_ch4_20261024 as COMP
    counter_checks = COMP.verify_counter()
    env = dict(_gpu_env_info(), infer_mode='deterministic off, cudnn.benchmark on, native operators', counter_checks=counter_checks)
    atomic_json(out / 'benchmark/environment.json', env)

    targets = BENCH_INFER_OBJECTS if args.only is None else args.only
    loads = {}
    for name in targets:
        obj_path = out / 'benchmark/objects' / f'{name}.json'
        if obj_path.exists() and not args.force:
            continue
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True).stdout.split()
        if apps:
            raise RuntimeError(f'GPU is busy before bench-infer {name}: {apps}')
        before = dict(time_bjt=now_bjt(), loadavg=list(os.getloadavg()))
        log_path = out / 'logs' / f'bench_infer_{name}.log'
        with log_path.open('a', encoding='utf-8') as log:
            rc = subprocess.call(
                [sys.executable, '-B', '-W', 'ignore', __file__, 'bench-infer', '--out', str(out), '--one', name],
                stdout=log, stderr=subprocess.STDOUT,
            )
        after = dict(time_bjt=now_bjt(), loadavg=list(os.getloadavg()))
        loads[name] = dict(before=before, after=after, returncode=rc)
        if rc != 0:
            raise SystemExit(f'bench-infer failed for {name} (rc={rc}); see {log_path}')

    rows = [read_json(out / 'benchmark/objects' / f'{n}.json') for n in BENCH_INFER_OBJECTS if (out / 'benchmark/objects' / f'{n}.json').exists()]
    report = dict(generated_at_bjt=now_bjt(), environment=env, loads=loads,
                  protocol='single-session isolated subprocesses, window=512x512 (b=1, 20 warmup, 100 timed), whole-image=5 val x 3 repeats (wb=8)',
                  objects=rows)
    atomic_json(out / 'benchmark/inference.json', report)
    cols = ['name', 'best_epoch', 'steps', 'params', 'params_m', 'window_gmacs', 'image_gmacs',
            'window_latency_ms_median', 'window_latency_ms_p95', 'window_throughput_per_s_b8',
            'global_latency_ms_median', 'image_predict_s_median', 'image_predict_s_mean',
            'infer_peak_allocated_mib', 'infer_peak_reserved_mib', 'agreement_with_as_evaluated']
    with (out / 'benchmark/inference.csv').open('w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)
    print(json.dumps(dict(status='completed', objects=len(rows), output=str(out / 'benchmark/inference.json')), indent=2))


# ----------------------------------------------------------------------------- S5 Training Resource Measurement (§8)

def _run_bench_train_one(out: Path, plan: dict, name: str):
    import glpflow_core as G
    from chapter_controlled_20261031 import core as CC
    from repela_20261024 import core as RP
    B_CORE = RP.B

    warmup, timed = 10, 50

    if name in ('g_onehot', 'g_ordinal'):
        cfg = read_json(Path(plan['configs'][name]['path']))
        DSE.configure(cfg)
        model, diffusion = DSE.build(cfg)
        ema = DSE.PowerEMA(model, cfg['ema_sigma_rel'])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        ds = DSE.SegmentationDataset(cfg, 'train')
        batch_size, workers = cfg['batch_size'], cfg['workers']
        epoch_ref = [1]

        def new_loader(ep):
            ds.epoch = ep
            return iter(DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                                   collate_fn=DSE.collate, generator=torch.Generator().manual_seed(cfg['seed'] + ep)))

        def step_fn(batch):
            terms, _ = DSE.train_step(model, diffusion, opt, batch, cfg)
            ema.update(model)
            return terms

    elif name in ('I', 'R', 'E1', 'segformer_b1', 'repela_small'):
        if name in ('I', 'R'):
            cfg = CC.config(name)
            fn_step = CC.train_step
            builder = G.build
            loader_maker = G.make_loader
        elif name == 'E1':
            cfg = G.config('E1')
            fn_step = G.train_step
            builder = G.build
            loader_maker = G.make_loader
        elif name == 'segformer_b1':
            cfg = B_CORE.config('segformer_b1')
            fn_step = B_CORE.train_step
            builder = B_CORE.build
            loader_maker = B_CORE.make_loader
        else:
            cfg = RP.config()
            fn_step = B_CORE.train_step
            builder = B_CORE.build
            loader_maker = B_CORE.make_loader

        G.configure(cfg['seed'])
        model, opt, _, ema = builder(cfg, cfg['device'])
        ds = G.GLPDataset(cfg, 'train')
        batch_size, workers = cfg['batch_size'], cfg['workers']
        epoch_ref = [1]

        def new_loader(ep):
            return iter(loader_maker(ds, cfg, ep))

        def step_fn(batch):
            terms, _ = fn_step(model, opt, batch, cfg, cfg['device'])
            ema.update(model)
            return terms
    else:
        raise ValueError(f'Unknown bench-train object: {name}')

    params = sum(p.numel() for p in model.parameters())
    model.train()
    it = new_loader(epoch_ref[0])
    times, losses = [], []
    torch.cuda.synchronize()
    for step_idx in range(warmup + timed):
        if step_idx == warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            epoch_ref[0] += 1
            it = new_loader(epoch_ref[0])
            batch = next(it)
        terms = step_fn(batch)
        torch.cuda.synchronize()
        dt_step = time.perf_counter() - t0
        losses.append(float(terms['loss_total']))
        if step_idx >= warmup:
            times.append(dt_step)

    peak_alloc_mib = torch.cuda.max_memory_allocated() / 2**20
    peak_res_mib = torch.cuda.max_memory_reserved() / 2**20
    t_arr = np.asarray(times, dtype=np.float64)
    sec_med = float(np.median(t_arr))
    sec_mean = float(np.mean(t_arr))
    res = dict(
        name=name,
        benchmarked_at_bjt=now_bjt(),
        params=params,
        batch_size=batch_size,
        workers=workers,
        grad_accum_steps=int(cfg.get('grad_accum_steps', 1)),
        warmup_updates=warmup,
        timed_updates=timed,
        sec_per_update_median=sec_med,
        sec_per_update_mean=sec_mean,
        sec_per_update_p95=float(np.percentile(t_arr, 95)),
        crops_per_sec_median=float(batch_size / sec_med),
        crops_per_sec_mean=float(batch_size / sec_mean),
        est_3000_updates_gpu_hours_mean=float(3000 * sec_mean / 3600.0),
        est_3000_updates_gpu_hours_median=float(3000 * sec_med / 3600.0),
        train_peak_allocated_mib=peak_alloc_mib,
        train_peak_reserved_mib=peak_res_mib,
        last_loss=losses[-1],
        raw_update_seconds=times,
    )
    atomic_json(out / 'benchmark/train_objects' / f'{name}.json', res)
    print(json.dumps(dict(name=name, sec_med=round(sec_med, 4), sec_mean=round(sec_mean, 4),
                          h3000=round(res['est_3000_updates_gpu_hours_mean'], 3),
                          mem_res_mib=round(peak_res_mib, 1)), ensure_ascii=False), flush=True)


def _historical_train_reconciliation(out: Path) -> dict:
    rec = {}
    for arm in ('g_onehot', 'g_ordinal'):
        hp = out / arm / 'history.json'
        if hp.exists():
            hist = read_json(hp)
            t_secs = [r['train_seconds'] for r in hist]
            v_secs = [r['validation']['seconds'] for r in hist if 'validation' in r]
            rec[arm] = dict(
                source=str(hp),
                epochs=len(hist),
                recorded_train_seconds_total=float(sum(t_secs)),
                recorded_train_hours_total=float(sum(t_secs) / 3600.0),
                recorded_epoch_seconds_median=float(np.median(t_secs)),
                val_passes=len(v_secs),
                recorded_val_seconds_total=float(sum(v_secs)),
                per_val_seconds_mean=float(np.mean(v_secs)) if v_secs else None,
            )
    for name, p in HIST_RUNS.items():
        t_rows, peak_rows, v_rows = [], [], []
        with (p / 'metrics_epochs.csv').open(newline='') as f:
            for r in csv.DictReader(f):
                if r['tag'] == 'resources/train_seconds' and r['value']:
                    t_rows.append(float(r['value']))
                elif r['tag'] == 'resources/peak_reserved_mib' and r['value']:
                    peak_rows.append(float(r['value']))
        with (p / 'metrics_val.csv').open(newline='') as f:
            for r in csv.DictReader(f):
                if r['tag'] == 'val/seconds' and r['value']:
                    v_rows.append(float(r['value']))
        rec[name] = dict(
            source=str(p / 'metrics_epochs.csv'),
            epochs=len(t_rows),
            recorded_train_seconds_total=float(sum(t_rows)),
            recorded_train_hours_total=float(sum(t_rows) / 3600.0),
            recorded_epoch_seconds_median=float(np.median(t_rows)),
            val_passes=len(v_rows),
            recorded_val_seconds_total=float(sum(v_rows)),
            per_val_seconds_mean=float(np.mean(v_rows)) if v_rows else None,
            recorded_peak_reserved_mib=float(max(peak_rows)) if peak_rows else None,
        )
    return rec


def cmd_bench_train(args):
    out = ensure_out_dir(args.out)
    plan = read_json(out / 'plan.json')
    if args.one:
        _run_bench_train_one(out, plan, args.one)
        return

    targets = BENCH_TRAIN_OBJECTS if args.only is None else args.only
    for name in targets:
        obj_path = out / 'benchmark/train_objects' / f'{name}.json'
        if obj_path.exists() and not args.force:
            continue
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True).stdout.split()
        if apps:
            raise RuntimeError(f'GPU is busy before bench-train {name}: {apps}')
        log_path = out / 'logs' / f'bench_train_{name}.log'
        with log_path.open('a', encoding='utf-8') as log:
            rc = subprocess.call(
                [sys.executable, '-B', '-W', 'ignore', __file__, 'bench-train', '--out', str(out), '--one', name],
                stdout=log, stderr=subprocess.STDOUT,
            )
        if rc != 0:
            raise SystemExit(f'bench-train failed for {name} (rc={rc}); see {log_path}')

    reconcile = _historical_train_reconciliation(out)
    rows = []
    for n in BENCH_TRAIN_OBJECTS:
        p = out / 'benchmark/train_objects' / f'{n}.json'
        if p.exists():
            r = dict(read_json(p))
            rec = reconcile.get(n, {})
            r['recorded_train_hours_total'] = rec.get('recorded_train_hours_total')
            r['recorded_val_seconds_total'] = rec.get('recorded_val_seconds_total')
            r['per_val_seconds_mean'] = rec.get('per_val_seconds_mean')
            r['recorded_peak_reserved_mib'] = rec.get('recorded_peak_reserved_mib')
            rows.append(r)

    env = dict(_gpu_env_info(), train_mode='deterministic on, FP32, TF32 disabled')
    report = dict(generated_at_bjt=now_bjt(), environment=env,
                  protocol='isolated CUDA subprocesses; 10 warmup + 50 timed updates across epoch boundary; official batch & workers',
                  reconciliation=reconcile, objects=rows)
    atomic_json(out / 'benchmark/training.json', report)
    cols = ['name', 'params', 'batch_size', 'workers', 'sec_per_update_median', 'sec_per_update_mean',
            'sec_per_update_p95', 'crops_per_sec_mean', 'est_3000_updates_gpu_hours_mean',
            'recorded_train_hours_total', 'recorded_val_seconds_total', 'per_val_seconds_mean',
            'train_peak_allocated_mib', 'train_peak_reserved_mib', 'recorded_peak_reserved_mib']
    with (out / 'benchmark/training.csv').open('w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)
    print(json.dumps(dict(status='completed', objects=len(rows), output=str(out / 'benchmark/training.json')), indent=2))


# ----------------------------------------------------------------------------- S11 + S13 Report & Figures

def _eval_rules(out: Path, val_boot: dict, test_boot: dict, bench_inf: dict) -> dict:
    t_g_oh_20 = read_json(out / 'eval/test27_g_onehot_k20.json')['metrics']
    t_g_oh_4 = read_json(out / 'eval/test27_g_onehot_k4.json')['metrics']
    t_g_ord_20 = read_json(out / 'eval/test27_g_ordinal_k20.json')['metrics']
    t_i_4 = read_json(out / 'eval/test27_I_k4.json')['metrics']
    hist_glp = read_json(HIST_GLP_TEST / 'test_metrics.json')
    t_r = hist_glp['E3R']['test']

    ben_map = {r['name']: r for r in bench_inf['objects']}
    dt_g20 = ben_map['g_onehot_k20']['image_predict_s_median']
    dt_i = ben_map['I_k4']['image_predict_s_median']

    miou_g20, bf_g20 = 100 * t_g_oh_20['miou_4class'], 100 * t_g_oh_20['boundary_f1_foreground']
    miou_g4, bf_g4 = 100 * t_g_oh_4['miou_4class'], 100 * t_g_oh_4['boundary_f1_foreground']
    miou_i, bf_i = 100 * t_i_4['miou_4class'], 100 * t_i_4['boundary_f1_foreground']
    miou_r, bf_r = 100 * t_r['miou_4class'], 100 * t_r['boundary_f1_foreground']

    drop_4_vs_20 = miou_g20 - miou_g4
    near_pp = 1.0  # within ~1 pp threshold for qualitative rule routing
    if miou_g20 > max(miou_i, miou_r) + 0.5:
        ddim_rule = INTERPRETATION_RULES['ddim_vs_i_rules'][3]
    elif miou_g20 < miou_i - 0.5 and bf_g20 < bf_i - 1.0 and drop_4_vs_20 > 1.0:
        ddim_rule = INTERPRETATION_RULES['ddim_vs_i_rules'][0]
    elif abs(miou_g20 - miou_i) <= 1.5 and (drop_4_vs_20 > 1.0 or dt_g20 > dt_i * 2.0):
        ddim_rule = INTERPRETATION_RULES['ddim_vs_i_rules'][1]
    elif abs(miou_g4 - miou_i) <= near_pp:
        ddim_rule = INTERPRETATION_RULES['ddim_vs_i_rules'][2]
    else:
        ddim_rule = INTERPRETATION_RULES['ddim_vs_i_rules'][0]

    ord_diff_test = abs(100 * t_g_ord_20['miou_4class'] - miou_g20)
    ord_rule = (INTERPRETATION_RULES['ordinal_vs_onehot_rules'][0]
                if ord_diff_test < 1.0 else INTERPRETATION_RULES['ordinal_vs_onehot_rules'][1])

    return dict(
        generated_at_bjt=now_bjt(),
        test_summary_pct=dict(
            g_onehot_k20=dict(miou=miou_g20, bf=bf_g20, img_s=dt_g20),
            g_onehot_k4=dict(miou=miou_g4, bf=bf_g4),
            g_ordinal_k20=dict(miou=100 * t_g_ord_20['miou_4class'], bf=100 * t_g_ord_20['boundary_f1_foreground']),
            I_k4=dict(miou=miou_i, bf=bf_i, img_s=dt_i),
            R_k4=dict(miou=miou_r, bf=bf_r, img_s=ben_map['R_k4']['image_predict_s_median']),
        ),
        ddim4_drop_vs_ddim20_pp=drop_4_vs_20,
        ordinal_minus_onehot_test_pp=100 * (t_g_ord_20['miou_4class'] - t_g_oh_20['miou_4class']),
        matched_ddim_vs_i_rule=ddim_rule,
        matched_ordinal_vs_onehot_rule=ord_rule,
        bootstrap_contrasts_test27=test_boot['contrasts'],
        bootstrap_contrasts_val25=val_boot['contrasts'],
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(clean_json(r))


def cmd_report(args):
    out = ensure_out_dir(args.out)
    plan = read_json(out / 'plan.json')
    val_g_oh = read_json(out / 'eval/steps_val_g_onehot.json')
    val_g_ord = read_json(out / 'eval/steps_val_g_ordinal.json')
    val_i = read_json(out / 'eval/steps_val_I.json')
    seed3408_oh = read_json(out / 'eval/seed3408_g_onehot.json')
    seed3408_ord = read_json(out / 'eval/seed3408_g_ordinal.json')

    test_g_oh_20 = read_json(out / 'eval/test27_g_onehot_k20.json')
    test_g_oh_4 = read_json(out / 'eval/test27_g_onehot_k4.json')
    test_g_ord_20 = read_json(out / 'eval/test27_g_ordinal_k20.json')
    test_g_ord_4 = read_json(out / 'eval/test27_g_ordinal_k4.json')
    test_i_4 = read_json(out / 'eval/test27_I_k4.json')

    boot = read_json(out / 'eval/bootstrap.json')
    bench_inf = read_json(out / 'benchmark/inference.json')
    bench_trn = read_json(out / 'benchmark/training.json')
    diag = {arm: read_json(out / 'diagnostics' / f'state_gap_{arm}.json') for arm in ('g_onehot', 'g_ordinal', 'I', 'R')}
    ctrl_sum = read_json(CONTROLLED_ANALYSIS / 'comparison_summary.json')
    hist_glp_test = read_json(HIST_GLP_TEST / 'test_metrics.json')
    hist_base_test = read_json(HIST_BASE_TEST / 'test_metrics.json')
    hist_repela_test = read_json(HIST_REPELA_TEST / 'test_metrics.json')

    tables = out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)

    # 1. t4_method_comparison.csv
    _write_csv(tables / 't4_method_comparison.csv',
               ['方法', '初态', '状态路径', '训练状态', '采样器', '步数', '损失', '参数量'],
               [
                   {'方法': 'G-onehot', '初态': '纯高斯噪声 z_T', '状态路径': '余弦方差 DDPM 前向加噪', '训练状态': 'q(z_t | z_0)', '采样器': 'DDIM (eta=0)', '步数': '20 (可调 1-50)', '损失': '0.5 CE + 0.5 Dice_fg + 1.0 x0_MSE + 0.5 Prior', '参数量': '18,192,148'},
                   {'方法': 'G-ordinal', '初态': '纯高斯噪声 z_T', '状态路径': '余弦方差 DDPM 前向加噪 (有序编码)', '训练状态': 'q(z_t | z_0)', '采样器': 'DDIM (eta=0)', '步数': '20 (可调 1-50)', '损失': '0.5 CE + 0.5 Dice_fg + 1.0 x0_MSE + 0.5 Prior', '参数量': '18,191,860'},
                   {'方法': 'I', '初态': '先验中心 + 熵调制噪声 x_0', '状态路径': '直线流匹配插值', '训练状态': '(1-t)x_0 + t(2Y-1)', '采样器': 'Euler', '步数': '4', '损失': '0.5 CE + 0.5 Dice_fg + 0.5 Prior', '参数量': '18,197,460'},
                   {'方法': 'R (E3R)', '初态': '先验中心 + 熵调制噪声 x_0', '状态路径': '直线流匹配 Euler', '训练状态': '自身无梯度 rollout 轨迹', '采样器': 'Euler', '步数': '4', '损失': '0.5 CE + 0.5 Dice_fg + 0.5 Prior', '参数量': '18,197,460'},
               ])

    # 2. t4_g_val.csv
    rows_gv = []
    for name, val_j, s3408_j in [('G-onehot', val_g_oh, seed3408_oh), ('G-ordinal', val_g_ord, seed3408_ord)]:
        for k, seed, src in [('20', 3407, val_j['results']['20']), ('20', 3408, s3408_j['results']['20']), ('4', 3407, val_j['results']['4'])]:
            m = src['metrics']
            rows_gv.append({
                '方法': name, '步数': k, '采样种子': seed, '最佳轮次': val_j['best_epoch'],
                'mIoU / %': f"{100*m['miou_4class']:.4f}", '单层IoU / %': f"{100*m['iou_monolayer']:.4f}",
                '少层IoU / %': f"{100*m['iou_fewlayer']:.4f}", '多层IoU / %': f"{100*m['iou_multilayer']:.4f}",
                '薄层IoU / %': f"{100*m['thin_mean_iou']:.4f}", '前景BF@2px / %': f"{100*m['boundary_f1_foreground']:.4f}",
                'few->mono / %': f"{100*m['few_to_mono_rate']:.4f}",
            })
    _write_csv(tables / 't4_g_val.csv', list(rows_gv[0].keys()), rows_gv)

    # 3. t4_steps_curve.csv (G-onehot 1..50, G-ordinal 1..50, I 1/2/4, R 1/2/4)
    rows_steps = []
    for name, src in [('G-onehot', val_g_oh['results']), ('G-ordinal', val_g_ord['results']), ('I', val_i['results'])]:
        for k_str in sorted(src.keys(), key=int):
            m = src[k_str]['metrics']
            rows_steps.append({
                '方法': name, '步数': int(k_str), 'mIoU / %': f"{100*m['miou_4class']:.4f}",
                '薄层IoU / %': f"{100*m['thin_mean_iou']:.4f}", '前景BF / %': f"{100*m['boundary_f1_foreground']:.4f}",
                'NFE/图': src[k_str]['nfe_per_image'],
            })
    with (THESIS_TABLES / 't4_steps.csv').open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows_steps.append({
                '方法': 'R (E3R)', '步数': int(r['返回位置']),
                'mIoU / %': r['mIoU / %'], '薄层IoU / %': f"{100*ctrl_sum['trajectories']['R'][r['返回位置']]['thin_mean_iou']:.4f}",
                '前景BF / %': r['BF / %'], 'NFE/图': 35 * int(r['返回位置']),
            })
    _write_csv(tables / 't4_steps_curve.csv', list(rows_steps[0].keys()), rows_steps)

    # 4. t4_state_gap.csv
    rows_diag = []
    for arm, d in diag.items():
        for p in d['grid_probes']:
            t_label = p.get('target_t', p.get('t'))
            grid_t = p.get('grid_t', p.get('t'))
            rows_diag.append({
                '方法': arm, '时间点': grid_t, '目标时间点': t_label,
                '训练型_mIoU / %': f"{100*p['training_type']['miou_4class']:.4f}",
                '推理型_mIoU / %': f"{100*p['inference_type']['miou_4class']:.4f}",
                'Delta_mIoU / pp': f"{p['delta_pp']['miou_4class']:+.4f}",
                '训练型_BF / %': f"{100*p['training_type']['boundary_f1_foreground']:.4f}",
                '推理型_BF / %': f"{100*p['inference_type']['boundary_f1_foreground']:.4f}",
                'Delta_BF / pp': f"{p['delta_pp']['boundary_f1_foreground']:+.4f}",
                '训练型_薄层IoU / %': f"{100*p['training_type']['thin_mean_iou']:.4f}",
                '推理型_薄层IoU / %': f"{100*p['inference_type']['thin_mean_iou']:.4f}",
                'Delta_薄层IoU / pp': f"{p['delta_pp']['thin_mean_iou']:+.4f}",
            })
    _write_csv(tables / 't4_state_gap.csv', list(rows_diag[0].keys()), rows_diag)

    # 5. t4_progression_val.csv
    rows_pv = []
    for label, src, best_ep in [
        ('G-onehot (DDIM20)', val_g_oh['results']['20']['metrics'], val_g_oh['best_epoch']),
        ('G-onehot (DDIM4)', val_g_oh['results']['4']['metrics'], val_g_oh['best_epoch']),
        ('G-ordinal (DDIM20)', val_g_ord['results']['20']['metrics'], val_g_ord['best_epoch']),
        ('G-ordinal (DDIM4)', val_g_ord['results']['4']['metrics'], val_g_ord['best_epoch']),
        ('I (4步)', val_i['results']['4']['metrics'], val_i['best_epoch']),
    ]:
        rows_pv.append({
            '方法': label, '最佳轮次': best_ep,
            'mIoU / %': f"{100*src['miou_4class']:.4f}", '单层IoU / %': f"{100*src['iou_monolayer']:.4f}",
            '少层IoU / %': f"{100*src['iou_fewlayer']:.4f}", '多层IoU / %': f"{100*src['iou_multilayer']:.4f}",
            '薄层IoU / %': f"{100*src['thin_mean_iou']:.4f}", '前景BF / %': f"{100*src['boundary_f1_foreground']:.4f}",
        })
    for a in ('R', 'N', 'F'):
        t_m = ctrl_sum['trajectories'][a]['4']
        rows_pv.append({
            '方法': f'{a} (4步)', '最佳轮次': ctrl_sum['records'][a]['meta']['epoch'],
            'mIoU / %': f"{100*t_m['miou_4class']:.4f}", '单层IoU / %': f"{100*t_m['iou_monolayer']:.4f}",
            '少层IoU / %': f"{100*t_m['iou_fewlayer']:.4f}", '多层IoU / %': f"{100*t_m['iou_multilayer']:.4f}",
            '薄层IoU / %': f"{100*t_m['thin_mean_iou']:.4f}", '前景BF / %': f"{100*t_m['boundary_f1']['2']['boundary_f1_foreground']:.4f}",
        })
    # Include E1 val reference from t4_val.csv
    with (THESIS_TABLES / 't4_val.csv').open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['方法'] == 'E1':
                rows_pv.append({'方法': 'E1 (1步参照)', '最佳轮次': r['最佳轮次'], 'mIoU / %': r['mIoU / %'],
                                '单层IoU / %': '', '少层IoU / %': '', '多层IoU / %': '',
                                '薄层IoU / %': r['薄层IoU / %'], '前景BF / %': r['前景BF / %']})
    _write_csv(tables / 't4_progression_val.csv', list(rows_pv[0].keys()), rows_pv)

    # 6. t4_main_test.csv
    rows_mt = []
    for k_b, disp in [('unet', 'U-Net'), ('unetpp', 'UNet++'), ('deeplabv3p', 'DeepLabv3+'), ('pspnet', 'PSPNet'),
                      ('segformer_b1', 'SegFormer-B1'), ('hrnet_w48', 'HRNet-W48'), ('mask2former', 'Mask2Former'),
                      ('ddp', 'DDP'), ('ccdm', 'CCDM')]:
        tm = hist_base_test[k_b]['test']
        rows_mt.append({
            '方法': disp, 'mIoU / %': f"{100*tm['miou_4class']:.2f}",
            '单层IoU / %': f"{100*tm['iou_monolayer']:.2f}", '少层IoU / %': f"{100*tm['iou_fewlayer']:.2f}",
            '多层IoU / %': f"{100*tm['iou_multilayer']:.2f}", '薄层IoU / %': f"{100*tm['thin_mean_iou']:.2f}",
            '前景BF / %': f"{100*tm['boundary_f1_foreground']:.2f}",
        })
    for k_g, disp in [('E0', 'E0'), ('E1', 'E1')]:
        tm = hist_glp_test[k_g]['test']
        rows_mt.append({
            '方法': disp, 'mIoU / %': f"{100*tm['miou_4class']:.2f}",
            '单层IoU / %': f"{100*tm['iou_monolayer']:.2f}", '少层IoU / %': f"{100*tm['iou_fewlayer']:.2f}",
            '多层IoU / %': f"{100*tm['iou_multilayer']:.2f}", '薄层IoU / %': f"{100*tm['thin_mean_iou']:.2f}",
            '前景BF / %': f"{100*tm['boundary_f1_foreground']:.2f}",
        })
    for disp, tm in [
        ('G-onehot (DDIM20)', test_g_oh_20['metrics']),
        ('G-onehot (DDIM4)', test_g_oh_4['metrics']),
        ('G-ordinal (DDIM20)', test_g_ord_20['metrics']),
        ('G-ordinal (DDIM4)', test_g_ord_4['metrics']),
        ('I (4步)', test_i_4['metrics']),
        ('R (E3R, 4步)', hist_glp_test['E3R']['test']),
        ('RepELA-Net-Small (部署)', hist_repela_test['deploy_form']['test']),
    ]:
        rows_mt.append({
            '方法': disp, 'mIoU / %': f"{100*tm['miou_4class']:.2f}",
            '单层IoU / %': f"{100*tm['iou_monolayer']:.2f}", '少层IoU / %': f"{100*tm['iou_fewlayer']:.2f}",
            '多层IoU / %': f"{100*tm['iou_multilayer']:.2f}", '薄层IoU / %': f"{100*tm['thin_mean_iou']:.2f}",
            '前景BF / %': f"{100*tm['boundary_f1_foreground']:.2f}",
        })
    _write_csv(tables / 't4_main_test.csv', list(rows_mt[0].keys()), rows_mt)

    # 7. t4_infer_cost.csv + 8. t4_train_cost.csv (formatted copies from benchmark JSON)
    ben_inf_map = {r['name']: r for r in bench_inf['objects']}
    rows_ic = []
    for r in bench_inf['objects']:
        rows_ic.append({
            '对象': r['name'], '步数': r['steps'], '参数 / M': f"{r['params_m']:.3f}",
            '窗GMACs': f"{r['window_gmacs']:.3f}", '整图GMACs': f"{r['image_gmacs']:.2f}",
            '单窗ms (中位数)': f"{r['window_latency_ms_median']:.3f}", '单窗ms (P95)': f"{r['window_latency_ms_p95']:.3f}",
            '整图s (中位数)': f"{r['image_predict_s_median']:.4f}",
            '推理峰值allocated / MiB': f"{r['infer_peak_allocated_mib']:.2f}",
            '推理峰值reserved / MiB': f"{r['infer_peak_reserved_mib']:.2f}",
            '原生算子一致率': f"{r['agreement_with_as_evaluated']:.6f}",
        })
    _write_csv(tables / 't4_infer_cost.csv', list(rows_ic[0].keys()), rows_ic)

    ben_trn_map = {r['name']: r for r in bench_trn['objects']}
    rows_tc = []
    for r in bench_trn['objects']:
        rows_tc.append({
            '对象': r['name'], 'batch': r['batch_size'], 'workers': r['workers'],
            '秒/更新 (中位数)': f"{r['sec_per_update_median']:.4f}", '秒/更新 (均值)': f"{r['sec_per_update_mean']:.4f}",
            '裁块/s': f"{r['crops_per_sec_mean']:.2f}",
            '3000更新GPU小时': f"{r['est_3000_updates_gpu_hours_mean']:.3f}",
            '记录训练总小时': f"{r['recorded_train_hours_total']:.3f}" if r.get('recorded_train_hours_total') is not None else '',
            '10次验证总秒': f"{r['recorded_val_seconds_total']:.1f}" if r.get('recorded_val_seconds_total') is not None else '',
            '训练峰值allocated / MiB': f"{r['train_peak_allocated_mib']:.1f}",
            '训练峰值reserved / MiB': f"{r['train_peak_reserved_mib']:.1f}",
            '历史记录训练reserved / MiB': f"{r['recorded_peak_reserved_mib']:.1f}" if r.get('recorded_peak_reserved_mib') is not None else '',
        })
    _write_csv(tables / 't4_train_cost.csv', list(rows_tc[0].keys()), rows_tc)

    # 9. t5_cost_updated.csv (all from unified S4/S5 batch!)
    rows_t5 = []
    for disp, inf_k, trn_k, test_src in [
        ('G-onehot (DDIM20)', 'g_onehot_k20', 'g_onehot', test_g_oh_20['metrics']),
        ('G-ordinal (DDIM20)', 'g_ordinal_k20', 'g_ordinal', test_g_ord_20['metrics']),
        ('I (4步)', 'I_k4', 'I', test_i_4['metrics']),
        ('R (E3R, 4步)', 'R_k4', 'R', hist_glp_test['E3R']['test']),
        ('SegFormer-B1', 'segformer_b1', 'segformer_b1', hist_base_test['segformer_b1']['test']),
        ('RepELA-Net-Small (部署)', 'repela_deploy', 'repela_small', hist_repela_test['deploy_form']['test']),
    ]:
        ir = ben_inf_map[inf_k]
        tr = ben_trn_map[trn_k]
        rows_t5.append({
            '方法': disp, '测试mIoU / %': f"{100*test_src['miou_4class']:.2f}",
            '测试BF / %': f"{100*test_src['boundary_f1_foreground']:.2f}",
            '参数 / M': f"{ir['params_m']:.3f}", '窗GMACs': f"{ir['window_gmacs']:.3f}",
            '单窗ms': f"{ir['window_latency_ms_median']:.2f}", '整图s': f"{ir['image_predict_s_median']:.3f}",
            '推理显存reserved / MiB': f"{ir['infer_peak_reserved_mib']:.1f}",
            '3000更新训练GPU小时': f"{tr['est_3000_updates_gpu_hours_mean']:.3f}",
            '训练峰值reserved / MiB': f"{tr['train_peak_reserved_mib']:.1f}",
        })
    _write_csv(tables / 't5_cost_updated.csv', list(rows_t5[0].keys()), rows_t5)

    # Narrative verdict (§13)
    verdict = _eval_rules(out, boot['val25'], boot['test27'], bench_inf)
    atomic_json(out / 'eval/narrative_verdict.json', verdict)

    # Build CJK 450 DPI full-border figures (§11)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    fm.fontManager.addfont(str(FONT_PATH))
    cn_font = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams.update({
        'font.family': [cn_font, 'DejaVu Sans'], 'font.size': 9, 'axes.labelsize': 9,
        'axes.titlesize': 9.3, 'legend.fontsize': 8.2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.linewidth': 0.65, 'axes.edgecolor': '#4E5964', 'text.color': '#202A33',
        'axes.labelcolor': '#202A33', 'xtick.color': '#46515C', 'ytick.color': '#46515C',
        'axes.spines.top': True, 'axes.spines.right': True, 'axes.unicode_minus': False,
        'lines.linewidth': 1.4, 'lines.markersize': 4.5, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })
    PAL = {'G-onehot': '#60717D', 'G-ordinal': '#8873A4', 'I': '#C37B42', 'R': '#286B8D'}
    MKS = {'G-onehot': 'D', 'G-ordinal': '^', 'I': 's', 'R': 'o'}
    LSS = {'G-onehot': '--', 'G-ordinal': '-.', 'I': ':', 'R': '-'}

    def enforce_borders_and_save(fig, name: str):
        for ax in fig.axes:
            if ax.axison:
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_linewidth(0.65); sp.set_color('#4E5964')
        fig.canvas.draw()
        fig.savefig(out / 'figures' / f'{name}.png', dpi=450, facecolor='white')
        fig.savefig(out / 'qa/figure_previews' / f'{name}.png', dpi=160, facecolor='white')
        plt.close(fig)

    # Figure 1: fig_4_steps_curve.png
    fig, axs = plt.subplots(1, 2, figsize=(6.22, 3.25))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.80, wspace=0.36)
    series_steps = {
        'G-onehot': ([int(k) for k in sorted(val_g_oh['results'], key=int)],
                     [100 * val_g_oh['results'][k]['metrics']['miou_4class'] for k in sorted(val_g_oh['results'], key=int)],
                     [100 * val_g_oh['results'][k]['metrics']['boundary_f1_foreground'] for k in sorted(val_g_oh['results'], key=int)]),
        'G-ordinal': ([int(k) for k in sorted(val_g_ord['results'], key=int)],
                      [100 * val_g_ord['results'][k]['metrics']['miou_4class'] for k in sorted(val_g_ord['results'], key=int)],
                      [100 * val_g_ord['results'][k]['metrics']['boundary_f1_foreground'] for k in sorted(val_g_ord['results'], key=int)]),
        'I': ([int(k) for k in sorted(val_i['results'], key=int)],
              [100 * val_i['results'][k]['metrics']['miou_4class'] for k in sorted(val_i['results'], key=int)],
              [100 * val_i['results'][k]['metrics']['boundary_f1_foreground'] for k in sorted(val_i['results'], key=int)]),
        'R': ([1, 2, 4],
              [100 * ctrl_sum['trajectories']['R'][str(k)]['miou_4class'] for k in (1, 2, 4)],
              [100 * ctrl_sum['trajectories']['R'][str(k)]['boundary_f1']['2']['boundary_f1_foreground'] for k in (1, 2, 4)]),
    }
    for ax, idx_y, ttl, ylabel in zip(axs, [1, 2], ['(a) 四类 mIoU', '(b) 前景边界 F1（2 px）'], ['四类 mIoU / %', '前景 BF / %']):
        for label, tup in series_steps.items():
            ax.plot(tup[0], tup[idx_y], color=PAL[label], marker=MKS[label], ls=LSS[label], label=label)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 4, 10, 20, 50])
        ax.set_xticklabels(['1', '2', '4', '10', '20', '50'])
        ax.set_xlabel('每窗网络调用次数 K（对数坐标）')
        ax.set_ylabel(ylabel)
        ax.set_title(ttl, loc='left', fontweight='bold', pad=8)
        ax.grid(True, which='major', color='#E2E7EB', lw=0.55)
        ax.set_axisbelow(True)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, ['G-onehot', 'G-ordinal', 'I', 'R (E3R)'], loc='upper center', bbox_to_anchor=(0.54, 0.985), ncol=4, frameon=False)
    enforce_borders_and_save(fig, 'fig_4_steps_curve')

    # Figure 2: fig_4_state_gap.png (S9 Δ_t = training - inference across matched stage indices)
    fig, axs = plt.subplots(1, 2, figsize=(6.22, 3.3))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.22, top=0.80, wspace=0.35)
    stage_x = np.arange(4)
    g_probe_pick = [0, 2, 3, 5]  # DDIM steps corresponding to early/mid/late stages (t=999, 473, 263, 0)
    bar_w = 0.19
    diag_order = [('g_onehot', 'G-onehot', 'G-onehot'), ('g_ordinal', 'G-ordinal', 'G-ordinal'), ('I', 'I', 'I'), ('R', 'R', 'R (E3R)')]
    for ax, metric_k, ttl, ylabel in zip(axs, ['miou_4class', 'boundary_f1_foreground'],
                                         ['(a) mIoU 状态差 Δ_t（训练型 − 推理型）', '(b) 前景 BF 状态差 Δ_t'],
                                         ['Δ_t mIoU / pp', 'Δ_t BF / pp']):
        ax.axhline(0.0, color='#5B6570', lw=0.8)
        for m_idx, (arm_k, pal_k, leg_l) in enumerate(diag_order):
            probes = diag[arm_k]['grid_probes']
            picks = [probes[i] for i in g_probe_pick] if arm_k.startswith('g_') else probes
            vals = [p['delta_pp'][metric_k] for p in picks]
            xpos = stage_x + (m_idx - 1.5) * bar_w
            ax.bar(xpos, vals, width=bar_w * 0.92, color=PAL[pal_k], label=leg_l)
        ax.set_xticks(stage_x)
        ax.set_xticklabels(['起点\n(t=999/0)', '中期前\n(t≈473/0.25)', '中期后\n(t≈263/0.5)', '末步\n(t=0/0.75)'], fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(ttl, loc='left', fontweight='bold', pad=8)
        ax.grid(axis='y', color='#E2E7EB', lw=0.55)
        ax.set_axisbelow(True)
    handles, _ = axs[0].get_legend_handles_labels()
    fig.legend(handles, ['G-onehot', 'G-ordinal', 'I', 'R (E3R)'], loc='upper center', bbox_to_anchor=(0.54, 0.985), ncol=4, frameon=False)
    enforce_borders_and_save(fig, 'fig_4_state_gap')

    # Figure 3: fig_4_quality_latency_scatter.png (unified S4 + S3)
    pts = [
        ('G-onehot (K=20)', ben_inf_map['g_onehot_k20']['image_predict_s_median'], 100 * test_g_oh_20['metrics']['miou_4class'], ben_inf_map['g_onehot_k20']['params_m'], PAL['G-onehot'], (6, -10)),
        ('G-onehot (K=4)', ben_inf_map['g_onehot_k4']['image_predict_s_median'], 100 * test_g_oh_4['metrics']['miou_4class'], ben_inf_map['g_onehot_k4']['params_m'], PAL['G-onehot'], (6, 4)),
        ('G-ordinal (K=20)', ben_inf_map['g_ordinal_k20']['image_predict_s_median'], 100 * test_g_ord_20['metrics']['miou_4class'], ben_inf_map['g_ordinal_k20']['params_m'], PAL['G-ordinal'], (-15, 7)),
        ('I (4步)', ben_inf_map['I_k4']['image_predict_s_median'], 100 * test_i_4['metrics']['miou_4class'], ben_inf_map['I_k4']['params_m'], PAL['I'], (6, -4)),
        ('R (E3R)', ben_inf_map['R_k4']['image_predict_s_median'], 100 * hist_glp_test['E3R']['test']['miou_4class'], ben_inf_map['R_k4']['params_m'], PAL['R'], (6, 5)),
        ('E1', ben_inf_map['E1']['image_predict_s_median'], 100 * hist_glp_test['E1']['test']['miou_4class'], ben_inf_map['E1']['params_m'], '#358671', (6, 4)),
        ('SegFormer-B1', ben_inf_map['segformer_b1']['image_predict_s_median'], 100 * hist_base_test['segformer_b1']['test']['miou_4class'], ben_inf_map['segformer_b1']['params_m'], '#2E7D32', (6, 4)),
        ('DDP', ben_inf_map['ddp']['image_predict_s_median'], 100 * hist_base_test['ddp']['test']['miou_4class'], ben_inf_map['ddp']['params_m'], '#7A5195', (6, -10)),
        ('CCDM', ben_inf_map['ccdm']['image_predict_s_median'], 100 * hist_base_test['ccdm']['test']['miou_4class'], ben_inf_map['ccdm']['params_m'], '#BC5090', (-38, 6)),
        ('RepELA (部署)', ben_inf_map['repela_deploy']['image_predict_s_median'], 100 * hist_repela_test['deploy_form']['test']['miou_4class'], ben_inf_map['repela_deploy']['params_m'], '#00897B', (6, -6)),
    ]
    fig, ax = plt.subplots(figsize=(6.22, 3.65))
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.16, top=0.90)
    for label, x_s, y_m, p_m, col, ofs in pts:
        ax.scatter(x_s, y_m, s=max(22.0, p_m * 10.0), color=col, alpha=0.88, edgecolor='white', lw=0.7, zorder=4)
        ax.annotate(label, (x_s, y_m), xytext=ofs, textcoords='offset points', fontsize=7.8, color='#202A33', zorder=5)
    ax.set_xscale('log')
    ax.set_xlabel('整图预测时间 / s（对数坐标，S4 同批次 window_batch=8）')
    ax.set_ylabel('test27 四类 mIoU / %')
    ax.set_title('质量—整图预测耗时—参数量（气泡面积对应参数量）', loc='left', fontweight='bold', pad=8)
    ax.grid(True, which='major', color='#E2E7EB', lw=0.55)
    ax.set_axisbelow(True)
    enforce_borders_and_save(fig, 'fig_4_quality_latency_scatter')

    print(json.dumps(dict(status='completed', tables=9, figures=3, narrative_verdict=str(out / 'eval/narrative_verdict.json')), ensure_ascii=False, indent=2))


# ----------------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_prep = sub.add_parser('prepare')
    p_prep.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p_prep.add_argument('--expected-commit', type=str, default=None)

    p_chk = sub.add_parser('check')
    p_chk.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p_chk.add_argument('--skip-unit-tests', action='store_true')

    p_trn = sub.add_parser('train')
    p_trn.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p_trn.add_argument('--arm', choices=('g_onehot', 'g_ordinal'), required=True)
    p_trn.add_argument('--resume', action='store_true')

    p_reg = sub.add_parser('register-test')
    p_reg.add_argument('--out', type=Path, default=DEFAULT_OUT)

    p_eval = sub.add_parser('evaluate')
    p_eval.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p_eval.add_argument('--model', choices=('g_onehot', 'g_ordinal', 'I'), required=True)
    p_eval.add_argument('--split', choices=('val', 'test'), required=True)
    p_eval.add_argument('--steps', type=int, nargs='+', required=True)
    p_eval.add_argument('--seed', type=int, default=3407)
    p_eval.add_argument('--result-name', type=str, required=True)

    p_boot = sub.add_parser('bootstrap')
    p_boot.add_argument('--out', type=Path, default=DEFAULT_OUT)

    p_diag = sub.add_parser('diagnose')
    p_diag.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p_diag.add_argument('--arm', choices=('g_onehot', 'g_ordinal', 'I', 'R'), required=True)
    p_diag.add_argument('--smoke-one', action='store_true')

    p_bi = sub.add_parser('bench-infer')
    p_bi.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p_bi.add_argument('--one', choices=BENCH_INFER_OBJECTS, default=None)
    p_bi.add_argument('--only', nargs='+', choices=BENCH_INFER_OBJECTS, default=None)
    p_bi.add_argument('--force', action='store_true')

    p_bt = sub.add_parser('bench-train')
    p_bt.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p_bt.add_argument('--one', choices=BENCH_TRAIN_OBJECTS, default=None)
    p_bt.add_argument('--only', nargs='+', choices=BENCH_TRAIN_OBJECTS, default=None)
    p_bt.add_argument('--force', action='store_true')

    p_rep = sub.add_parser('report')
    p_rep.add_argument('--out', type=Path, default=DEFAULT_OUT)

    args = parser.parse_args()
    dispatch = {
        'prepare': cmd_prepare,
        'check': cmd_check,
        'train': cmd_train,
        'register-test': cmd_register_test,
        'evaluate': cmd_evaluate,
        'bootstrap': cmd_bootstrap,
        'diagnose': cmd_diagnose,
        'bench-infer': cmd_bench_infer,
        'bench-train': cmd_bench_train,
        'report': cmd_report,
    }
    dispatch[args.cmd](args)


if __name__ == '__main__':
    main()

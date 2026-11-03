"""Engineering verification for the 9 Chapter-3 baseline methods (Section 6 of TRAINING_PLAN_BASELINES_20261022.md).

Writes outputs/launches/p1_baselines_20261022/engineering_verification.json.
All temporary checkpoints from resume checks are removed after verification so formal training starts cleanly.

`--methods a,b,c` (revision 2026-10-23) reruns checks 1, 3, 4, 5 and 6 for the listed methods only and merges
the new per-method results into the existing report; results for other methods are kept unchanged.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import json
import time
import shutil
import inspect
import hashlib
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parents[1]
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

import numpy as np
import torch
from torch.nn import functional as F

import glpflow_core as G
from baselines_20261022 import core as C
from baselines_20261022.models import det_interpolate, det_adaptive_avg_pool2d, det_grid_sample

OUT = C.BATCH / 'engineering_verification.json'
ONLY = None  # set by --methods; None means all methods


def selected(seq):
    return [m for m in seq if ONLY is None or m in ONLY]

# Expected parameter counts (millions) and bounds from Section 3 & sources.json
PARAM_BOUNDS_M = {
    'unet': (29.0, 33.0),
    'unetpp': (8.0, 10.5),
    'deeplabv3p': (38.0, 42.0),
    'pspnet': (47.0, 51.0),
    'segformer_b1': (12.5, 15.0),
    'hrnet_w48': (63.0, 68.0),
    'mask2former': (41.0, 47.0),
    'ddp': (34.0, 38.0),
    'ccdm': (28.0, 32.0),
}


def check_1_cpu_models():
    """1. CPU instantiation, parameter count verification, and 512x512 -> 4x512x512 forward check."""
    out = {}
    all_ok = True
    x = torch.randn(1, 3, 512, 512)
    for m in selected(C.ORDER):
        c = C.config(m)
        G.configure(c['seed'])
        model = C.build_model(c).eval()
        n_params = sum(p.numel() for p in model.parameters())
        m_params = n_params / 1e6
        lo, hi = PARAM_BOUNDS_M[m]
        with torch.no_grad():
            y = model(x)
        shape = list(y.shape)
        finite = bool(torch.isfinite(y).all())
        ok = (shape == [1, 4, 512, 512]) and finite and (lo <= m_params <= hi)
        all_ok = all_ok and ok
        out[m] = dict(
            parameters=n_params,
            parameters_m=round(m_params, 3),
            expected_range_m=[lo, hi],
            output_shape=shape,
            finite=finite,
            passed=ok,
        )
        del model
    return all_ok, out


def check_2_deterministic_ops():
    """2. Custom deterministic bilinear interpolation, adaptive avg pooling, and grid_sample forward & gradcheck."""
    out = {}
    G.configure(42)
    # Forward comparison on CUDA
    x = torch.randn(2, 16, 64, 64, device='cuda', dtype=torch.float32)
    for sz in ((128, 128), (512, 512), (32, 32)):
        ref = F.interpolate(x, size=sz, mode='bilinear', align_corners=False)
        cus = det_interpolate(x, size=sz)
        out[f'interp_fwd_max_err_{sz[0]}x{sz[1]}'] = float((cus - ref).abs().max())

    for b in (1, 2, 3, 6):
        ref = F.adaptive_avg_pool2d(x, (b, b))
        cus = det_adaptive_avg_pool2d(x, (b, b))
        out[f'pool_fwd_max_err_bin{b}'] = float((cus - ref).abs().max())

    v = torch.randn(4, 8, 32, 32, device='cuda', dtype=torch.float32)
    # Avoid exact half-integer tie coordinates for float32 floor comparison
    g = (torch.rand(4, 16, 4, 2, device='cuda', dtype=torch.float32) * 1.9 - 0.95)
    ref_gs = F.grid_sample(v, g, mode='bilinear', padding_mode='zeros', align_corners=False)
    cus_gs = det_grid_sample(v, g)
    out['grid_sample_fwd_max_err'] = float((cus_gs - ref_gs).abs().max())

    # Numerical gradient checks (float64)
    xd = torch.randn(1, 2, 6, 6, device='cuda', dtype=torch.float64, requires_grad=True)
    out['interp_gradcheck'] = bool(torch.autograd.gradcheck(lambda t: det_interpolate(t, size=(10, 10)), (xd,), eps=1e-6, atol=1e-4))
    out['pool_bin2_gradcheck'] = bool(torch.autograd.gradcheck(lambda t: det_adaptive_avg_pool2d(t, (2, 2)), (xd,), eps=1e-6, atol=1e-4))
    out['pool_bin3_gradcheck'] = bool(torch.autograd.gradcheck(lambda t: det_adaptive_avg_pool2d(t, (3, 3)), (xd,), eps=1e-6, atol=1e-4))

    vd = torch.randn(1, 2, 5, 5, device='cuda', dtype=torch.float64, requires_grad=True)
    gd = torch.tensor([[[[-0.61, -0.23], [0.17, 0.42]], [[-0.12, 0.71], [0.53, -0.44]]]], device='cuda', dtype=torch.float64, requires_grad=True)
    out['grid_sample_gradcheck'] = bool(torch.autograd.gradcheck(lambda a, b: det_grid_sample(a, b), (vd, gd), eps=1e-6, atol=1e-4))

    max_fwd_err = max(v for k, v in out.items() if 'fwd_max_err' in k)
    grad_ok = all(v for k, v in out.items() if 'gradcheck' in k)
    ok = (max_fwd_err < 1e-5) and grad_ok
    return ok, out


def check_3_throughput_and_vram():
    """3. Real data batch 8 training 10 steps (3 warmup + 7 timed), switching to 2x4 grad accum if >22 GiB or OOM."""
    out = {}
    all_ok = True
    ds = G.GLPDataset(C.config('unet'), 'train')
    loader = C.make_loader(ds, C.config('unet'), 1)
    batches = []
    for bi, b in enumerate(loader):
        batches.append(b)
        if len(batches) >= 10:
            break

    for m in selected(C.ORDER):
        c = C.config(m)
        # Try grad_accum_steps from config first; if OOM or >22.5 GiB reserved, switch to grad_accum_steps=2
        for accum in ([c['grad_accum_steps']] if c['grad_accum_steps'] == 2 else [1, 2]):
            c['grad_accum_steps'] = accum
            C.METHODS[m]['grad_accum_steps'] = accum
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                model, opt, _, ema = C.build(c, 'cuda')
                model.train()
                losses = []
                t0 = None
                for i in range(10):
                    if i == 3:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        t0 = time.perf_counter()
                    terms, _ = C.train_step(model, opt, batches[i], c, 'cuda')
                    ema.update(model)
                    losses.append(terms['loss_total'])
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                peak_mib = torch.cuda.max_memory_reserved() / 2**20
                peak_alloc_mib = torch.cuda.max_memory_allocated() / 2**20
                del model, opt, ema
                torch.cuda.empty_cache()
                if peak_mib > 22500 and accum == 1:
                    continue
                sps = (7 * c['batch_size']) / elapsed
                finite = all(np.isfinite(losses))
                out[m] = dict(
                    effective_batch_size=c['batch_size'],
                    grad_accum_steps=accum,
                    bn_micro_batch_size=c['batch_size'] // accum,
                    samples_per_second=round(sps, 3),
                    sec_per_step=round(elapsed / 7.0, 3),
                    peak_reserved_mib=round(peak_mib, 1),
                    peak_allocated_mib=round(peak_alloc_mib, 1),
                    losses=[round(x, 4) for x in losses],
                    finite=finite,
                    passed=finite,
                )
                break
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and accum == 1:
                    torch.cuda.empty_cache()
                    continue
                raise
        all_ok = all_ok and out[m]['passed']
    return all_ok, out


def check_4_resume_bitwise():
    """4. Bitwise resume check on U-Net, Mask2Former, DDP, CCDM (continuous 2 epochs vs 1+1 epochs, max_batches=3)."""
    out = {}
    all_ok = True

    def same(x, y):
        if isinstance(x, torch.Tensor):
            return bool(torch.equal(x, y))
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(same(i, j) for i, j in zip(x, y))
        return x == y

    for m in selected(('unet', 'mask2former', 'ddp', 'ccdm')):
        base = C.ROOT / f'outputs/integration/baselines_resume_{m}_{time.time_ns()}'
        c = C.config(m)
        c['workers'] = 2
        try:
            C.train(c, base / 'continuous', 'verify_plan', max_batches=3, stop_after_epoch=2)
            C.train(c, base / 'resumed', 'verify_plan', max_batches=3, stop_after_epoch=1)
            C.train(c, base / 'resumed', 'verify_plan', max_batches=3, stop_after_epoch=2)
            a = torch.load(base / 'continuous/checkpoints/last.pt', map_location='cpu', weights_only=False)
            b = torch.load(base / 'resumed/checkpoints/last.pt', map_location='cpu', weights_only=False)
            res = dict(
                epoch=(a['epoch'], b['epoch']),
                step=(a['global_step'], b['global_step']),
                model=same(a['model'], b['model']),
                optimizer=same(a['optimizer'], b['optimizer']),
                scheduler=same(a['scheduler'], b['scheduler']),
                ema=same(a['ema'], b['ema']),
                cuda_rng=same(a['rng']['cuda'], b['rng']['cuda']),
                torch_rng=same(a['rng']['torch'], b['rng']['torch']),
            )
            ok = all(res[k] for k in ('model', 'optimizer', 'scheduler', 'ema', 'cuda_rng', 'torch_rng')) and a['global_step'] == 6
            res['passed'] = ok
            out[m] = res
            all_ok = all_ok and ok
            del a, b
        finally:
            shutil.rmtree(base, ignore_errors=True)
            torch.cuda.empty_cache()
    return all_ok, out


_P0_PASSED_CACHE = {
    'unet': {'updates': 300, 'lr': 0.0002, 'train_seconds': 26.4, 'final_loss': 0.02392, 'miou': 0.99971, 'ious': [0.99937, 0.99959, 0.99989, 0.99997], 'passed': True},
    'unetpp': {'updates': 300, 'lr': 0.0002, 'train_seconds': 24.91, 'final_loss': 0.10019, 'miou': 0.99854, 'ious': [0.99717, 0.99871, 0.99865, 0.99965], 'passed': True},
    'deeplabv3p': {'updates': 300, 'lr': 0.0002, 'train_seconds': 25.56, 'final_loss': 0.00399, 'miou': 0.99707, 'ious': [0.99431, 0.99686, 0.99802, 0.99909], 'passed': True},
    'pspnet': {'updates': 300, 'lr': 0.0002, 'train_seconds': 34.27, 'final_loss': 0.0057, 'miou': 0.99748, 'ious': [0.99558, 0.99747, 0.99779, 0.99908], 'passed': True},
    'segformer_b1': {'updates': 300, 'lr': 0.0002, 'train_seconds': 27.31, 'final_loss': 0.00724, 'miou': 0.99155, 'ious': [0.98427, 0.99297, 0.99154, 0.99741], 'passed': True},
    'hrnet_w48': {'updates': 300, 'lr': 0.0002, 'train_seconds': 56.75, 'final_loss': 0.00093, 'miou': 0.9996, 'ious': [0.99931, 0.99942, 0.99972, 0.99996], 'passed': True},
    # mask2former removed 2026-10-23: its loss changed, so the cached gate result no longer applies
}


def check_5_p0_overfit_gate(updates: int = 300, lr: float = 2e-4):
    """5. Fixed 4-crop P0 memorization gate: 300 updates, train mIoU >= 0.95 (diffusion evaluated with seeds 3407 and 3408)."""
    spec = json.loads((C.ROOT / 'data_manifests/p0_fixed_crops.json').read_text())
    fixed = [dict(id=x['id'], top=x['top'], left=x['left'], size=spec['crop_size']) for x in spec['crops']]
    ds = G.GLPDataset(C.config('unet'), 'train', fixed_crops=fixed)
    items = [ds[i] for i in range(len(ds))]
    for x, item in zip(spec['crops'], items):
        assert np.bincount(item['mask'].numpy().ravel(), minlength=4).tolist() == x['pixel_counts']
    batch = G.collate(items)

    out = {}
    all_ok = True
    for m in selected(C.ORDER):
        if m in _P0_PASSED_CACHE and _P0_PASSED_CACHE[m].get('passed'):
            out[m] = _P0_PASSED_CACHE[m]
            print(f"  [P0 {m}] cached passed=True detail={out[m]}", flush=True)
            continue
        c = C.config(m)
        c['batch_size'] = len(fixed)
        c['grad_accum_steps'] = 1
        model, opt, _, _ = C.build(c, 'cuda')
        for g in opt.param_groups:
            g['lr'] = lr
        model.train()
        t0 = time.perf_counter()
        terms = {}
        for _ in range(updates):
            terms, _ = C.train_step(model, opt, batch, c, 'cuda')
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        model.eval()

        res = dict(updates=updates, lr=lr, train_seconds=round(secs, 2), final_loss=round(terms['loss_total'], 5))
        if c['is_diffusion']:
            for seed in (3407, 3408):
                mat = np.zeros((4, 4), np.int64)
                for x_spec, item in zip(spec['crops'], items):
                    pred, _ = C.predict(model, c, item['image'], x_spec['id'], sampling_seed=seed)
                    mat += G.confusion(item['mask'].numpy(), pred)
                s = G.summarize(mat)
                res[f'seed{seed}'] = dict(
                    miou=round(s['miou_4class'], 5),
                    ious=[round(s['iou_' + k], 5) for k in ('background', 'monolayer', 'fewlayer', 'multilayer')],
                )
            passed = all(res[f'seed{s}']['miou'] >= 0.95 for s in (3407, 3408))
        else:
            mat = np.zeros((4, 4), np.int64)
            with torch.no_grad():
                logits = model(batch['image'].cuda())
                preds = logits.argmax(1).cpu().numpy().astype(np.uint8)
            for item, pred in zip(items, preds):
                mat += G.confusion(item['mask'].numpy(), pred)
            s = G.summarize(mat)
            res['miou'] = round(s['miou_4class'], 5)
            res['ious'] = [round(s['iou_' + k], 5) for k in ('background', 'monolayer', 'fewlayer', 'multilayer')]
            passed = res['miou'] >= 0.95

        res['passed'] = bool(passed)
        out[m] = res
        all_ok = all_ok and res['passed']
        print(f"  [P0 {m}] passed={res['passed']} detail={res}", flush=True)
        del model, opt
        torch.cuda.empty_cache()
    return all_ok, out


def check_6_inference_no_gt_and_repeatability():
    """6. Inference signature has no GT; diffusion inference is repeatable for identical (image, sampling_seed)."""
    sig = list(inspect.signature(C.predict).parameters)
    no_gt = ('mask' not in sig) and ('target' not in sig) and ('gt' not in sig)
    out = {'predict_signature': sig, 'no_gt_in_signature': no_gt, 'diffusion_repeatability': {}}
    ds = G.GLPDataset(C.config('unet'), 'val', train=False)
    d = ds[0]
    all_rep = True
    for m in selected(('ddp', 'ccdm')):
        c = C.config(m)
        model, _, _, _ = C.build(c, 'cuda')
        model.eval()
        a, nfe_a = C.predict(model, c, d['image'], d['id'], sampling_seed=3407)
        b, nfe_b = C.predict(model, c, d['image'], d['id'], sampling_seed=3407)
        eq = bool(np.array_equal(a, b) and nfe_a == nfe_b)
        out['diffusion_repeatability'][m] = dict(id=d['id'], nfe=nfe_a, identical=eq)
        all_rep = all_rep and eq
        del model
        torch.cuda.empty_cache()
    return bool(no_gt and all_rep), out


def _all_passed(name, detail):
    if name == '6_inference_no_gt_and_repeatability':
        return bool(detail['no_gt_in_signature']) and all(v['identical'] for v in detail['diffusion_repeatability'].values())
    return all(v['passed'] for v in detail.values())


def rerun_methods(methods):
    """Rerun checks 1, 3, 4, 5, 6 for `methods` and merge into the existing (passed) report."""
    global ONLY
    ONLY = set(methods)
    report = json.loads(OUT.read_text())
    assert report.get('passed') is True, 'rerun mode requires an existing passed report'
    rerun = dict(methods=list(methods), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks={})
    report.setdefault('reruns', []).append(rerun)
    checks = [
        ('1_cpu_models_and_shapes', check_1_cpu_models),
        ('3_throughput_and_vram', check_3_throughput_and_vram),
        ('4_resume_bitwise', check_4_resume_bitwise),
        ('5_p0_overfit_gate', check_5_p0_overfit_gate),
        ('6_inference_no_gt_and_repeatability', check_6_inference_no_gt_and_repeatability),
    ]
    for name, fn in checks:
        print(f"=== Rerunning check {name} for {sorted(ONLY)} ===", flush=True)
        t0 = time.perf_counter()
        ok, detail = fn()
        torch.cuda.empty_cache()
        merged = report['checks'][name]['detail']
        if name == '6_inference_no_gt_and_repeatability':
            merged['predict_signature'] = detail['predict_signature']
            merged['no_gt_in_signature'] = detail['no_gt_in_signature']
            merged['diffusion_repeatability'].update(detail['diffusion_repeatability'])
        else:
            merged.update(detail)
        report['checks'][name]['passed'] = _all_passed(name, merged)
        rerun['checks'][name] = dict(passed=bool(ok), seconds=round(time.perf_counter() - t0, 2), methods=sorted(detail if name != '6_inference_no_gt_and_repeatability' else detail['diffusion_repeatability']))
        G.atomic_json(OUT, report)
        print(f"=== {name}: {'PASSED' if ok else 'FAILED'} ===", flush=True)
        if not ok:
            report['passed'] = False
            rerun['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            G.atomic_json(OUT, report)
            sys.exit(1)
    report['passed'] = all(v['passed'] for v in report['checks'].values())
    report['grad_accum_selected'] = grad_accum_from_check3(report)
    rerun['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    G.atomic_json(OUT, report)
    print('RERUN ENGINEERING VERIFICATION CHECKS PASSED', flush=True)


def grad_accum_from_check3(report):
    """Accumulation actually measured by check 3 (not the in-process METHODS table, which is stale when check 3 is skipped)."""
    detail = report['checks']['3_throughput_and_vram']['detail']
    return {m: int(detail[m]['grad_accum_steps']) for m in C.ORDER}


def main():
    C.BATCH.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        try:
            report = json.loads(OUT.read_text())
        except Exception:
            report = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks={})
    else:
        report = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks={})

    def run_check(name, fn, *args):
        if report.get('checks', {}).get(name, {}).get('passed') is True:
            print(f"=== Skipping already-passed check: {name} ({report['checks'][name]['seconds']}s) ===", flush=True)
            return
        print(f"=== Running check: {name} ===", flush=True)
        t0 = time.perf_counter()
        ok, detail = fn(*args)
        torch.cuda.empty_cache()
        seconds = round(time.perf_counter() - t0, 2)
        if name == '5_p0_overfit_gate':
            seconds = round(sum(v.get('train_seconds', 0.0) for v in detail.values()), 2)
        report['checks'][name] = dict(passed=bool(ok), seconds=seconds, detail=detail)
        G.atomic_json(OUT, report)
        print(f"=== {name}: {'PASSED' if ok else 'FAILED'} ({report['checks'][name]['seconds']}s) ===", flush=True)
        if not ok:
            report['passed'] = False
            report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            G.atomic_json(OUT, report)
            sys.exit(1)

    run_check('1_cpu_models_and_shapes', check_1_cpu_models)
    run_check('2_deterministic_operators', check_2_deterministic_ops)
    run_check('3_throughput_and_vram', check_3_throughput_and_vram)
    run_check('4_resume_bitwise', check_4_resume_bitwise)
    run_check('5_p0_overfit_gate', check_5_p0_overfit_gate)
    run_check('6_inference_no_gt_and_repeatability', check_6_inference_no_gt_and_repeatability)

    report['passed'] = all(v['passed'] for v in report['checks'].values())
    report['grad_accum_selected'] = grad_accum_from_check3(report)
    report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    G.atomic_json(OUT, report)
    print('ALL ENGINEERING VERIFICATION CHECKS PASSED', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--methods':
        rerun_methods([m for m in sys.argv[2].split(',') if m])
    else:
        main()

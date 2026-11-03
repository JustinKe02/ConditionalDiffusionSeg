"""Engineering verification for RepELA-Net-Small under the unified protocol (2026-10-24).

Writes outputs/launches/p1_ch4_extension_20261024/engineering_verification_repela.json. Checks:
  1. source copies byte-identical to PhysicalNet; parameter count; 512x512 -> 4x512x512 forward;
  2. deterministic substitutions vs native PhysicalNet operators (random input and real val windows);
  3. reparameterization: train-form vs deploy-form outputs after real training steps;
  4. batch-8 throughput and peak memory on real training batches;
  5. bitwise resume (2 epochs vs 1+1 epochs, 3 batches each) through baselines core.train;
  6. P0 fixed-crop memorization gate (300 updates, train mIoU >= 0.95);
  7. prediction signature takes no ground truth.
Temporary resume runs are deleted afterwards.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import copy
import json
import time
import shutil
import inspect
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parents[1]
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

import numpy as np
import torch

import glpflow_core as G
from repela_20261024 import core as R
from repela_20261024.model import RepELASmall, native_ops

B = R.B
OUT = R.BATCH / 'engineering_verification_repela.json'


def same(x, y):
    if isinstance(x, torch.Tensor):
        return bool(torch.equal(x, y))
    if isinstance(x, np.ndarray):
        return bool(np.array_equal(x, y))
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(same(i, j) for i, j in zip(x, y))
    return x == y


def val_windows(n=8):
    c = R.config()
    d = G.GLPDataset(c, 'val', train=False)[0]
    img = d['image']
    sz, st = c['crop_size'], c['stride']
    coords = [(y, x) for y in B.positions(img.shape[1], sz, st) for x in B.positions(img.shape[2], sz, st)][:n]
    return torch.stack([img[:, y:y + sz, x:x + sz] for y, x in coords])


def train_batches(n=10):
    c = R.config()
    ds = G.GLPDataset(c, 'train')
    out = []
    for b in B.make_loader(ds, c, 1):
        out.append(b)
        if len(out) >= n:
            break
    return out


def check_1_source_and_shapes():
    hashes = {f: dict(source=G.digest(R.SOURCE_DIR / f), copy=G.digest(R.COPY_DIR / f)) for f in R.SOURCE_FILES}
    identical = all(v['source'] == v['copy'] for v in hashes.values())
    c = R.config()
    G.configure(c['seed'])
    model = R.build_model(c).eval()
    n = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        y = model(torch.randn(1, 3, 512, 512))
    ok = identical and list(y.shape) == [1, 4, 512, 512] and bool(torch.isfinite(y).all()) and 2.0e6 <= n <= 2.3e6
    return ok, dict(file_sha256=hashes, byte_identical=identical, parameters=n, parameters_m=round(n / 1e6, 4),
                    replaced_global_pools=model.replaced_pools, output_shape=list(y.shape))


def check_2_substitutions():
    c = R.config()
    G.configure(c['seed'])
    det = R.build_model(c).cuda().eval()
    ref = RepELASmall(4, deterministic_pools=False).cuda().eval()
    ref.load_state_dict(det.state_dict())
    out = {}
    with torch.no_grad():
        for name, x in (('random', torch.randn(4, 3, 512, 512, device='cuda')), ('val_windows', val_windows().cuda())):
            a = det(x)
            with native_ops():
                b = ref(x)
            out[name] = dict(max_abs_diff=float((a - b).abs().max()), argmax_agreement=float((a.argmax(1) == b.argmax(1)).float().mean()))
    ok = all(v['max_abs_diff'] < 1e-4 for v in out.values())
    return ok, out


def check_3_reparameterization(steps=20):
    c = R.config()
    model, opt, _, _ = B.build(c, 'cuda')
    model.train()
    for b in train_batches(steps // 2) * 2:
        B.train_step(model, opt, b, c, 'cuda')
    model.eval()
    deploy = copy.deepcopy(model).switch_to_deploy().eval()
    x = val_windows().cuda()
    with torch.no_grad():
        a = model(x)
        b = deploy(x)
    res = dict(updates=steps, params_train_form=sum(p.numel() for p in model.parameters()),
               params_deploy_form=sum(p.numel() for p in deploy.parameters()),
               max_abs_diff=float((a - b).abs().max()), max_abs_logit=float(a.abs().max()),
               argmax_agreement=float((a.argmax(1) == b.argmax(1)).float().mean()))
    ok = res['max_abs_diff'] < 1e-4 * max(1.0, res['max_abs_logit']) and res['argmax_agreement'] > 0.9999
    return ok, res


def check_4_throughput():
    c = R.config()
    batches = train_batches(10)
    torch.cuda.empty_cache()
    model, opt, _, ema = B.build(c, 'cuda')
    model.train()
    losses, t0 = [], None
    for i, b in enumerate(batches):
        if i == 3:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        terms, _ = B.train_step(model, opt, b, c, 'cuda')
        ema.update(model)
        losses.append(terms['loss_total'])
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    res = dict(batch_size=c['batch_size'], samples_per_second=round(7 * c['batch_size'] / el, 3), sec_per_step=round(el / 7, 4),
               peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 1),
               peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1), losses=[round(v, 4) for v in losses])
    ok = all(np.isfinite(losses)) and res['peak_reserved_mib'] < 22500
    return ok, res


def check_5_resume():
    base = R.ROOT / f'outputs/integration/repela_resume_{time.time_ns()}'
    c = R.config()
    c['workers'] = 2
    try:
        B.train(c, base / 'continuous', 'verify_plan', max_batches=3, stop_after_epoch=2)
        B.train(c, base / 'resumed', 'verify_plan', max_batches=3, stop_after_epoch=1)
        B.train(c, base / 'resumed', 'verify_plan', max_batches=3, stop_after_epoch=2)
        a = torch.load(base / 'continuous/checkpoints/last.pt', map_location='cpu', weights_only=False)
        b = torch.load(base / 'resumed/checkpoints/last.pt', map_location='cpu', weights_only=False)
        res = dict(step=(a['global_step'], b['global_step']), model=same(a['model'], b['model']),
                   optimizer=same(a['optimizer'], b['optimizer']), scheduler=same(a['scheduler'], b['scheduler']),
                   ema=same(a['ema'], b['ema']), cuda_rng=same(a['rng']['cuda'], b['rng']['cuda']),
                   torch_rng=same(a['rng']['torch'], b['rng']['torch']),
                   python_rng=same(a['rng']['python'], b['rng']['python']),
                   numpy_rng=same(a['rng']['numpy'], b['rng']['numpy']))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    ok = all(res[k] for k in ('model', 'optimizer', 'scheduler', 'ema', 'cuda_rng', 'torch_rng', 'python_rng', 'numpy_rng')) and res['step'] == (6, 6)
    return ok, res


def check_6_p0(updates=300, lr=2e-4):
    spec = json.loads((R.ROOT / 'data_manifests/p0_fixed_crops.json').read_text())
    fixed = [dict(id=x['id'], top=x['top'], left=x['left'], size=spec['crop_size']) for x in spec['crops']]
    c = R.config()
    ds = G.GLPDataset(c, 'train', fixed_crops=fixed)
    items = [ds[i] for i in range(len(ds))]
    for x, item in zip(spec['crops'], items):
        assert np.bincount(item['mask'].numpy().ravel(), minlength=4).tolist() == x['pixel_counts']
    batch = G.collate(items)
    c['batch_size'] = len(fixed)
    model, opt, _, _ = B.build(c, 'cuda')
    for g in opt.param_groups:
        g['lr'] = lr
    model.train()
    t0 = time.perf_counter()
    for _ in range(updates):
        terms, _ = B.train_step(model, opt, batch, c, 'cuda')
    torch.cuda.synchronize()
    model.eval()
    with torch.no_grad():
        preds = model(batch['image'].cuda()).argmax(1).cpu().numpy().astype(np.uint8)
    mat = sum(G.confusion(item['mask'].numpy(), p) for item, p in zip(items, preds))
    s = G.summarize(mat)
    res = dict(updates=updates, lr=lr, train_seconds=round(time.perf_counter() - t0, 2), final_loss=round(terms['loss_total'], 5),
               miou=round(s['miou_4class'], 5), ious=[round(s['iou_' + k], 5) for k in ('background', 'monolayer', 'fewlayer', 'multilayer')])
    return res['miou'] >= 0.95, res


def check_7_no_gt():
    sig = list(inspect.signature(B.predict).parameters)
    ok = not any(k in sig for k in ('mask', 'target', 'gt'))
    return ok, dict(predict_signature=sig)


def main():
    R.BATCH.mkdir(parents=True, exist_ok=True)
    report = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks={})
    checks = [('1_source_and_shapes', check_1_source_and_shapes), ('2_deterministic_substitutions', check_2_substitutions),
              ('3_reparameterization', check_3_reparameterization), ('4_throughput_and_vram', check_4_throughput),
              ('5_resume_bitwise', check_5_resume), ('6_p0_overfit_gate', check_6_p0), ('7_inference_no_gt', check_7_no_gt)]
    for name, fn in checks:
        print(f'=== {name} ===', flush=True)
        t0 = time.perf_counter()
        ok, detail = fn()
        torch.cuda.empty_cache()
        report['checks'][name] = dict(passed=bool(ok), seconds=round(time.perf_counter() - t0, 2), detail=detail)
        G.atomic_json(OUT, report)
        print(f"=== {name}: {'PASSED' if ok else 'FAILED'} {json.dumps(detail)[:600]}", flush=True)
        if not ok:
            report['passed'] = False
            G.atomic_json(OUT, report)
            sys.exit(1)
    report['passed'] = True
    report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    G.atomic_json(OUT, report)
    print('REPELA ENGINEERING VERIFICATION PASSED', flush=True)


if __name__ == '__main__':
    main()

"""Engineering verification for the GLP-Flow batch. Writes BATCH/engineering_verification.json.

Checks: unit maths, no ground truth in prediction, bitwise resume, P0 fixed-crop memorization gate.
Integration/P0 weights are kept separately and never used for the effect training.
"""
import os, sys, json, time, inspect, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import glpflow_core as G

OUT = G.BATCH/'engineering_verification.json'


def unit_checks():
    out = {}
    y = torch.rand(2, 4, 8, 8)*2-1; x = torch.rand(2, 4, 8, 8)*2-1
    t0 = torch.tensor([.25, .5]); h = torch.tensor([.125, .25])
    # constant endpoint prediction: two half steps must imply the same endpoint
    xm = G.euler(x, y, t0, h); xe = G.euler(xm, y, t0+h, h)
    implied = x+((1-t0)/(2*h))[:, None, None, None]*(xe-x)
    out['shortcut_constant_endpoint_max_error'] = float((implied-y).abs().max())
    # path endpoints
    for name, tt in (('t0', 0.), ('t1', 1.)):
        xt = (1-tt)*x+tt*y
        out[f'path_{name}_error'] = float((xt-(x if tt == 0 else y)).abs().max())
    # the last Euler step lands on the predicted endpoint
    out['final_step_error'] = float((G.euler(x, y, torch.tensor([.75, .75]), torch.tensor([.25, .25]))-y).abs().max())
    out['level_mapping'] = {k: G.level_for_steps(k) for k in (1, 2, 4, 8)}
    out['ema_gamma_sigma0.10'] = G.PowerEMA.solve(.10)
    p = torch.full((1, 4, 2, 2), .25); u = G.entropy_map(p)
    out['uniform_entropy_is_one'] = float(u.min())
    c3 = G.config('E3'); eps = torch.randn(1, 4, 2, 2)
    conf = torch.zeros(1, 4, 2, 2); conf[:, 1] = 1
    out['confident_source_ignores_noise'] = float((G.make_source(c3, conf, G.entropy_map(conf), eps)-(2*conf-1)).abs().max())
    out['predict_signature'] = list(inspect.signature(G.predict).parameters)
    ok = (out['shortcut_constant_endpoint_max_error'] < 1e-5 and out['path_t0_error'] == 0 and out['path_t1_error'] == 0
          and out['final_step_error'] < 1e-6 and abs(out['ema_gamma_sigma0.10']-6.94) < .05
          and abs(out['uniform_entropy_is_one']-1) < 1e-5 and out['confident_source_ignores_noise'] < 1e-3
          and 'mask' not in out['predict_signature'] and 'target' not in out['predict_signature'])
    return ok, out


def prediction_repeatability():
    c = G.config('E3'); model, _, _, _ = G.build(c); model.eval()
    d = G.GLPDataset(c, 'val', train=False)[0]
    a, nfe = G.predict(model, c, d['image'], d['thumb'], d['id'])
    b, _ = G.predict(model, c, d['image'], d['thumb'], d['id'])
    return bool(np.array_equal(a, b)), dict(id=d['id'], nfe=nfe, identical=bool(np.array_equal(a, b)))


def resume_check():
    base = G.ROOT/f'outputs/integration/glpflow_resume_{time.time_ns()}'
    c = G.config('E3')
    G.train(c, base/'continuous', 'verify', max_batches=3, stop_after_epoch=2)
    G.train(c, base/'resumed', 'verify', max_batches=3, stop_after_epoch=1)
    G.train(c, base/'resumed', 'verify', max_batches=3, stop_after_epoch=2)
    a = torch.load(base/'continuous/checkpoints/last.pt', map_location='cpu', weights_only=False)
    b = torch.load(base/'resumed/checkpoints/last.pt', map_location='cpu', weights_only=False)
    def same(x, y):
        if isinstance(x, torch.Tensor): return torch.equal(x, y)
        if isinstance(x, dict): return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)): return len(x) == len(y) and all(same(i, j) for i, j in zip(x, y))
        return x == y
    res = dict(directory=str(base), epoch=(a['epoch'], b['epoch']), step=(a['global_step'], b['global_step']),
               model=same(a['model'], b['model']), optimizer=same(a['optimizer'], b['optimizer']),
               scheduler=same(a['scheduler'], b['scheduler']), ema=same(a['ema'], b['ema']),
               cuda_rng=same(a['rng']['cuda'], b['rng']['cuda']), torch_rng=same(a['rng']['torch'], b['rng']['torch']))
    ok = all(res[k] for k in ('model', 'optimizer', 'scheduler', 'ema', 'cuda_rng', 'torch_rng')) and a['global_step'] == 6
    return ok, res


def p0_gate(arm, updates=300, lr=2e-4):
    spec = json.loads((G.ROOT/'data_manifests/p0_fixed_crops.json').read_text())
    fixed = [dict(id=x['id'], top=x['top'], left=x['left'], size=spec['crop_size']) for x in spec['crops']]
    c = G.config(arm); c['batch_size'] = len(fixed)
    model, opt, _, _ = G.build(c)
    for g in opt.param_groups: g['lr'] = lr
    ds = G.GLPDataset(c, 'train', fixed_crops=fixed); items = [ds[i] for i in range(len(ds))]
    for x, item in zip(spec['crops'], items):
        assert np.bincount(item['mask'].numpy().ravel(), minlength=4).tolist() == x['pixel_counts']
    batch = G.collate(items); model.train(); t0 = time.perf_counter()
    for _ in range(updates): terms, _ = G.train_step(model, opt, batch, c, 'cuda')
    model.eval(); result = dict(arm=arm, updates=updates, lr=lr, final_terms=terms, train_seconds=time.perf_counter()-t0)
    for seed in (3407, 3408):
        m = np.zeros((4, 4), np.int64)
        for x, item in zip(spec['crops'], items):
            ctx = model.global_context([item['thumb'].cuda()])
            gen = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f"{x['id']}:{seed}".encode()).digest()[:8], 'little') % (2**63-1))
            eps = torch.randn((1, 4, spec['crop_size'], spec['crop_size']), generator=gen).cuda()
            logits, _ = G.infer_windows(model, c, item['image'][None].cuda(), item['box'][None], ctx, eps, c['sampling_steps'])
            m += G.confusion(item['mask'].numpy(), logits.argmax(1)[0].cpu().numpy().astype(np.uint8))
        s = G.summarize(m); ious = [s['iou_'+k] for k in ('background', 'monolayer', 'fewlayer', 'multilayer')]
        result[f'seed{seed}'] = dict(miou=s['miou_4class'], ious=ious)
    passed = all(result[f'seed{s}']['miou'] >= .90 and min(result[f'seed{s}']['ious']) >= .75 for s in (3407, 3408))
    result['passed'] = passed
    return passed, result


if __name__ == '__main__':
    G.BATCH.mkdir(parents=True, exist_ok=True)
    report = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks={})
    def run(name, fn, *a):
        t = time.perf_counter(); ok, detail = fn(*a); torch.cuda.empty_cache()
        report['checks'][name] = dict(passed=bool(ok), seconds=time.perf_counter()-t, detail=detail)
        print(name, 'PASSED' if ok else 'FAILED', json.dumps(detail, default=str)[:600], flush=True)
        G.atomic_json(OUT, report)
    run('unit', unit_checks)
    run('prediction_repeatability_no_gt', prediction_repeatability)
    run('resume_bitwise', resume_check)
    run('p0_gate_E3', p0_gate, 'E3')
    run('p0_informational_E2', p0_gate, 'E2')
    report['benchmark_batch_selection'] = dict(
        arm='E3', note='interactive 2026-10-21 end-to-end loop incl. loader (12 workers), 10 timed steps after 4 warmup',
        samples_per_second={'4': 4.45, '8': 4.94, '12': 5.12}, peak_reserved_mib={'4': 8798, '8': 17108, '12': 23716},
        selected=8, reason='batch 12 leaves <1 GiB headroom on 24 GiB; batch 8 is 11% faster than 4 with ~7 GiB headroom',
        lr_rule='2e-4 at batch 4 scaled by sqrt(2) to 2.8e-4 for batch 8')
    required = [k for k in report['checks'] if not k.startswith('p0_informational')]
    report['passed'] = all(report['checks'][k]['passed'] for k in required)
    report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    G.atomic_json(OUT, report)
    print('ALL REQUIRED PASSED' if report['passed'] else 'VERIFICATION FAILED', flush=True)
    sys.exit(0 if report['passed'] else 1)

"""Sampler-aligned GLP-Flow (E3R): conditional on the 2026-10-22 diagnosis confirming a state gap.

Only the training state changes relative to E3: each sample draws k in {0..K-1}; its state x_k is produced
by the model's own no-grad K-step sampler from the same prior/uncertainty source (no GT interpolation),
with self-conditioning exactly as at inference; the gradient step is taken at t=k/K. Shortcut is off
because training now uses the inference grid directly. Architecture, data, budget and validation equal E3.
Usage: (coordinator)  python launch: --verify | --worker | --secondary | (no args = coordinator)
"""
import os, sys, json, time, fcntl, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import glpflow_core as G

BATCH = G.ROOT/'outputs/launches/p1_glpflow_rollout_20261022'
DIAG = G.ROOT/'outputs/analysis/glpflow_diagnosis_20261022/diagnosis.json'
OLD = G.ROOT/'outputs/launches/p1_glpflow_20261021'
PLAN = BATCH/'plan.json'
PY = sys.executable
FROZEN = ['work/glpflow_core.py', 'work/glpflow_rollout_20261022.py', 'work/diagnose_glpflow_20261022.py',
          'outputs/analysis/glpflow_diagnosis_20261022/diagnosis.json', 'outputs/launches/p1_glpflow_rollout_20261022/engineering_verification.json',
          'data_manifests/source_dataset.json', 'data_manifests/splits/train.txt', 'data_manifests/splits/val.txt']
GATES = dict(miou_E3R_minus_E1=.005, thin_mean_E3R_minus_E1=.010, boundary_E3R_minus_E1_min=-.005,
             few_to_mono_E3R_below_E1=True, miou_E3R_minus_E3=.008, seed3408_E3R_above_E1=True,
             statistical_significance_claim=False)


def config():
    c = G.config('E3')
    c.update(arm='E3R', training_state='sampler_rollout_k4_no_grad', shortcut_fraction=0.,
             self_condition='inference-matched (previous rollout step)', self_condition_prob=None)
    return c


def train_step_rollout(model, opt, batch, c, device):
    image, mask, boxes = batch['image'].to(device), batch['mask'].to(device), batch['box']
    b, s = image.shape[0], image.shape[-1]; K = c['sampling_steps']
    feats = model.local(image)
    ctx = model.global_context([t.to(device) for t in batch['thumb']])
    p, umap, gtok, gpos, _ = G.window_conditions(model, c, ctx, boxes, s, device)
    prior_loss = image.new_zeros(())
    for (_, pl, _), soft in zip(ctx, batch['soft']):
        soft = soft.to(device)
        prior_loss = prior_loss+(-(soft*pl.log_softmax(0)).sum(0).mean()+G.dice_loss(pl[None], soft.argmax(0)[None]))/2
    prior_loss = prior_loss/b
    lv = torch.full((b,), G.level_for_steps(K), device=device, dtype=torch.long)
    k = torch.randint(K, (b,), device=device)
    x = G.make_source(c, p, umap, torch.randn(b, 4, s, s, device=device)); sc = torch.zeros_like(p)
    with torch.no_grad():  # model-generated states; GT never enters the input state
        for j in range(K-1):
            idx = (k > j).nonzero().flatten()
            if not len(idx): break
            t = torch.full((len(idx),), j/K, device=device)
            out = model.denoise([f[idx] for f in feats], x[idx], p[idx], umap[idx], sc[idx], t, lv[idx],
                                gtok[idx], gpos[idx], [boxes[int(i)] for i in idx]).softmax(1)
            x[idx] = G.euler(x[idx], 2*out-1, t, torch.full_like(t, 1/K)); sc[idx] = out
    t = k.float()/K
    logits = model.denoise(feats, x.detach(), p, umap, sc.detach(), t, lv, gtok, gpos, boxes)
    terms = dict(loss_ce=F.cross_entropy(logits, mask, reduction='none').mean(), loss_dice_fg=G.dice_loss(logits, mask), loss_prior=prior_loss)
    total = .5*terms['loss_ce']+.5*terms['loss_dice_fg']+c['prior_loss_weight']*prior_loss
    terms['loss_total'] = total
    if not torch.isfinite(total): raise FloatingPointError('Nonfinite loss')
    opt.zero_grad(set_to_none=True); total.backward()
    grad = nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'], error_if_nonfinite=True); opt.step()
    terms['mean_k'] = k.float().mean()
    return {n: float(v.detach()) for n, v in terms.items()}, float(grad)


G.train_step = train_step_rollout  # G.train and the verification helpers resolve train_step at call time


def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def status(**kw):
    f = BATCH/'status.json'; s = json.loads(f.read_text()) if f.exists() else {}
    s.update(kw, updated=now()); G.atomic_json(f, s)


def verify():
    import verify_glpflow_20261021 as V
    V.G.config = lambda arm, _orig=V.G.config: config() if arm == 'E3R' else _orig(arm)
    report = dict(started=now(), checks={})
    c = config(); model, opt, _, ema = G.build(c)
    ds = G.GLPDataset(c, 'train'); it = iter(G.make_loader(ds, c, 1)); torch.cuda.reset_peak_memory_stats(); n = 0
    for i in range(10):
        bt = next(it)
        if i == 3: torch.cuda.synchronize(); t0 = time.perf_counter(); n = 0
        terms, _ = train_step_rollout(model, opt, bt, c, 'cuda'); ema.update(model); n += bt['image'].shape[0]
    torch.cuda.synchronize()
    report['checks']['smoke_b8'] = dict(passed=all(np.isfinite(v) for v in terms.values()), samples_per_second=n/(time.perf_counter()-t0),
                                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, terms=terms)
    del model, opt, ema, it; torch.cuda.empty_cache()
    # bitwise resume with the rollout step
    base = G.ROOT/f'outputs/integration/glpflow_rollout_resume_{time.time_ns()}'
    G.train(c, base/'continuous', 'verify', max_batches=3, stop_after_epoch=2)
    G.train(c, base/'resumed', 'verify', max_batches=3, stop_after_epoch=1)
    G.train(c, base/'resumed', 'verify', max_batches=3, stop_after_epoch=2)
    a = torch.load(base/'continuous/checkpoints/last.pt', map_location='cpu', weights_only=False)
    bb = torch.load(base/'resumed/checkpoints/last.pt', map_location='cpu', weights_only=False)
    same = lambda x, y: all(torch.equal(x[k], y[k]) for k in x)
    report['checks']['resume_bitwise'] = dict(passed=same(a['model'], bb['model']) and same(a['ema']['model'], bb['ema']['model'])
                                              and torch.equal(a['rng']['cuda'][0], bb['rng']['cuda'][0]), directory=str(base))
    ok, detail = V.p0_gate('E3R'); report['checks']['p0_gate_E3R'] = dict(passed=ok, detail=detail)
    report['passed'] = all(v['passed'] for v in report['checks'].values()); report['finished'] = now()
    G.atomic_json(BATCH/'engineering_verification.json', report); print(json.dumps(report, default=str)[:1500], flush=True)
    sys.exit(0 if report['passed'] else 1)


def guard(plan):
    for p, h in plan['frozen_sha256'].items(): assert G.digest(G.ROOT/p) == h, f'frozen file changed: {p}'


def worker():
    plan = json.loads(PLAN.read_text()); guard(plan); c = plan['config']
    print(G.train(c, plan['run'], G.digest(PLAN), guard_fn=lambda: guard(plan)), flush=True)


def secondary():
    plan = json.loads(PLAN.read_text()); guard(plan); c = plan['config']; G.configure(c['seed'])
    state = torch.load(Path(plan['run'])/'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model = G.GLPFlowNet(c).cuda(); model.load_state_dict(state['model'])
    metrics, mats, _ = G.evaluate(model, c, sampling_seed=3408)
    G.atomic_json(BATCH/'secondary_E3R_3408.json', dict(epoch=state['epoch'], metrics=metrics))
    np.savez_compressed(BATCH/'secondary_E3R_3408.npz', **mats)


def report(plan):
    old = json.loads((OLD/'comparison_summary.json').read_text())['results']
    run = Path(plan['run']); best = json.loads((run/'best_val_metrics.json').read_text())
    r, e1, e3 = best['metrics'], old['E1']['metrics'], old['E3']['metrics']
    s3408 = json.loads((BATCH/'secondary_E3R_3408.json').read_text())['metrics']
    checks = dict(miou_E3R_minus_E1=r['miou_4class']-e1['miou_4class'] >= GATES['miou_E3R_minus_E1'],
                  thin_mean_E3R_minus_E1=r['thin_mean_iou']-e1['thin_mean_iou'] >= GATES['thin_mean_E3R_minus_E1'],
                  boundary_E3R_minus_E1=r['boundary_f1_foreground']-e1['boundary_f1_foreground'] >= GATES['boundary_E3R_minus_E1_min'],
                  few_to_mono_E3R_below_E1=r['few_to_mono_rate'] < e1['few_to_mono_rate'],
                  miou_E3R_minus_E3=r['miou_4class']-e3['miou_4class'] >= GATES['miou_E3R_minus_E3'],
                  seed3408_E3R_above_E1=s3408['miou_4class'] > e1['miou_4class'])
    out = dict(E3R=dict(best_epoch=best['epoch'], metrics=r, seed3408=s3408), E1=e1, E3=e3, checks=checks,
               screening_passed=all(checks.values()), gates=GATES, scope=plan['scope'])
    G.atomic_json(BATCH/'comparison_summary.json', out)
    rows = [('E1', old['E1']['best_epoch'], e1), ('E3', old['E3']['best_epoch'], e3), ('E3R', best['epoch'], r)]
    lines = ['# Sampler-aligned E3R completed', '', '| Arm | Best epoch | mIoU % | Mono % | Few % | Thin mean % | FG boundary F1 % | few→mono % |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for n, e, z in rows:
        lines.append(f"| {n} | {e} | {100*z['miou_4class']:.2f} | {100*z['iou_monolayer']:.2f} | {100*z['iou_fewlayer']:.2f} | {100*z['thin_mean_iou']:.2f} | {100*z['boundary_f1_foreground']:.2f} | {100*z['few_to_mono_rate']:.2f} |")
    lines += ['', f"E3R 3408 mIoU: {100*s3408['miou_4class']:.2f}%", f"Screen passed: {out['screening_passed']}", '', '```', json.dumps(checks, indent=1), '```',
              '', 'Single training seed 42; val25 selection; E1/E3 reused from p1_glpflow_20261021; no test.']
    (BATCH/'final_analysis.md').write_text('\n'.join(lines)+'\n'); return out


def call(args, log):
    with open(BATCH/log, 'a') as f:
        return subprocess.call([PY, '-B', '-W', 'ignore', __file__, *args], stdout=f, stderr=subprocess.STDOUT)


def coordinator():
    BATCH.mkdir(parents=True, exist_ok=True)
    lock = open(G.ROOT/'outputs/.experiments.lock', 'w'); fcntl.flock(lock, fcntl.LOCK_EX)  # waits for the diagnosis
    status(stage='waiting_diagnosis', pid=os.getpid())
    diag = json.loads(DIAG.read_text())
    if 'decision' not in diag: status(stage='failed_diagnosis_incomplete'); sys.exit(2)
    if not diag['decision']['reproduces_recorded_E3_k4']: status(stage='failed_diagnosis_not_reproducible'); sys.exit(2)
    if not diag['decision']['confirmed']:
        status(stage='stopped_hypothesis_not_confirmed', decision=diag['decision']); return
    if not (BATCH/'engineering_verification.json').exists():
        status(stage='engineering_verification')
        if call(['--verify'], 'engineering_verification.log'): status(stage='failed_verification'); sys.exit(1)
    if not PLAN.exists():
        G.atomic_json(PLAN, dict(batch_id='p1_glpflow_rollout_20261022', created=now(), config=config(), gates=GATES,
                                 run=str(G.RUN_ROOT/'p1_glpflow_rollout_20261022_e3r'), diagnosis_decision=diag['decision'],
                                 frozen_sha256={p: G.digest(G.ROOT/p) for p in FROZEN},
                                 scope='seed42 single run; train120/val25 incl. m109/m31; same budget as E3 (100 ep, b8, 3000 updates); E1/E3 reused; no test; report and stop'))
    plan = json.loads(PLAN.read_text()); guard(plan)
    st = Path(plan['run'])/'status.json'
    if not (st.exists() and json.loads(st.read_text()).get('status') == 'completed'):
        status(stage='training_E3R', started_arm=now())
        if call(['--worker'], 'train_E3R.log'): status(stage='failed_E3R'); sys.exit(1)
    if not (BATCH/'secondary_E3R_3408.json').exists():
        status(stage='secondary_E3R_3408')
        if call(['--secondary'], 'secondary_E3R.log'): status(stage='failed_secondary'); sys.exit(1)
    out = report(plan); guard(plan); status(stage='completed', screening_passed=out['screening_passed'], finished=now())


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    {'--verify': verify, '--worker': worker, '--secondary': secondary}.get(arg, coordinator)()

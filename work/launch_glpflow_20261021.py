"""Coordinator for the authorized GLP-Flow batch: E0 -> E1 -> E2 -> E3, 3408 checks, report, stop.

Rerun the same command to resume; completed arms are not retrained, last.pt decides the position.
"""
import os, sys, json, time, fcntl, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import glpflow_core as G

PY = sys.executable
PLAN = G.BATCH/'plan.json'
ORDER = ['E0', 'E1', 'E2', 'E3']
FROZEN = ['work/glpflow_core.py', 'work/verify_glpflow_20261021.py', 'work/launch_glpflow_20261021.py',
          'outputs/launches/p1_glpflow_20261021/engineering_verification.json',
          'data_manifests/source_dataset.json', 'data_manifests/splits/train.txt', 'data_manifests/splits/val.txt']
GATES = dict(miou_E3_minus_E1=.005, thin_mean_E3_minus_E1=.010, few_to_mono_E3_below_E1=True,
             miou_E3_minus_E0=.015, miou_E3_minus_E2=.008, boundary_E3_minus_E1_min=-.005,
             seed3408_E3_above_E1_and_E2=True, statistical_significance_claim=False)


def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def status(**kw):
    s = json.loads((G.BATCH/'status.json').read_text()) if (G.BATCH/'status.json').exists() else {}
    s.update(kw, updated=now()); G.atomic_json(G.BATCH/'status.json', s)


def freeze():
    if PLAN.exists(): return json.loads(PLAN.read_text())
    v = json.loads((G.BATCH/'engineering_verification.json').read_text())
    assert v['passed'], 'engineering verification must pass before freezing'
    plan = dict(batch_id='p1_glpflow_20261021', created=now(), order=ORDER, gates=GATES,
                configs={a: G.config(a) for a in ORDER}, runs={a: str(G.RUN_ROOT/f'p1_glpflow_20261021_{a.lower()}') for a in ORDER},
                frozen_sha256={p: G.digest(G.ROOT/p) for p in FROZEN},
                scope='seed42 single training run per arm; original train120/val25 incl. m109/m31; no test; report and stop',
                budget='100 epochs x 240 crops = 24000 crops, batch 8, 3000 updates per arm; validation every 10 epochs (EMA weights)',
                selection='best by global 4-class val mIoU (EMA sigma_rel 0.10), sampling seed 3407; 3408 same-best check for E2/E3')
    G.atomic_json(PLAN, plan); return plan


def guard(plan):
    for p, h in plan['frozen_sha256'].items():
        assert G.digest(G.ROOT/p) == h, f'frozen file changed: {p}'


def worker(arm):
    plan = json.loads(PLAN.read_text()); guard(plan); c = plan['configs'][arm]
    G.configure(c['seed'])
    print(G.train(c, plan['runs'][arm], G.digest(PLAN), guard_fn=lambda: guard(plan)), flush=True)


def secondary(arm):
    import torch
    plan = json.loads(PLAN.read_text()); guard(plan); c = plan['configs'][arm]; run = Path(plan['runs'][arm])
    G.configure(c['seed']); state = torch.load(run/'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model = G.GLPFlowNet(c).cuda(); model.load_state_dict(state['model'])
    metrics, mats, _ = G.evaluate(model, c, sampling_seed=3408)
    G.atomic_json(G.BATCH/f'secondary_{arm}_3408.json', dict(epoch=state['epoch'], metrics=metrics))
    np.savez_compressed(G.BATCH/f'secondary_{arm}_3408.npz', **mats)


def report(plan):
    res = {}
    for a in ORDER:
        run = Path(plan['runs'][a]); best = json.loads((run/'best_val_metrics.json').read_text())
        z = np.load(run/'best_val_confusion.npz')
        keep = [i for i, s in enumerate(z['ids']) if str(s) != 'm109']
        no109 = G.classification_metrics(z['per_image'][keep].sum(0))
        meta = json.loads((run/'run_metadata.json').read_text())
        res[a] = dict(best_epoch=best['epoch'], metrics=best['metrics'], parameters=meta['parameters'],
                      descriptive_without_m109=dict(miou=no109['miou_4class'], thin_mean_iou=no109['thin_mean_iou']))
        f = G.BATCH/f'secondary_{a}_3408.json'
        if f.exists(): res[a]['seed3408'] = json.loads(f.read_text())['metrics']
    m = {a: res[a]['metrics'] for a in ORDER}
    s3408 = lambda a: res[a].get('seed3408', m[a])['miou_4class']
    checks = dict(
        miou_E3_minus_E1=m['E3']['miou_4class']-m['E1']['miou_4class'] >= GATES['miou_E3_minus_E1'],
        thin_mean_E3_minus_E1=m['E3']['thin_mean_iou']-m['E1']['thin_mean_iou'] >= GATES['thin_mean_E3_minus_E1'],
        few_to_mono_E3_below_E1=m['E3']['few_to_mono_rate'] < m['E1']['few_to_mono_rate'],
        miou_E3_minus_E0=m['E3']['miou_4class']-m['E0']['miou_4class'] >= GATES['miou_E3_minus_E0'],
        miou_E3_minus_E2=m['E3']['miou_4class']-m['E2']['miou_4class'] >= GATES['miou_E3_minus_E2'],
        boundary_E3_minus_E1=m['E3']['boundary_f1_foreground']-m['E1']['boundary_f1_foreground'] >= GATES['boundary_E3_minus_E1_min'],
        seed3408_E3_above_E1_and_E2=s3408('E3') > s3408('E1') and s3408('E3') > s3408('E2'))
    out = dict(results=res, checks=checks, screening_passed=all(checks.values()), gates=GATES, scope=plan['scope'])
    G.atomic_json(G.BATCH/'comparison_summary.json', out)
    lines = ['# GLP-Flow batch completed', '', '| Arm | Best epoch | mIoU % | Mono % | Few % | Thin mean % | FG boundary F1 % | few→mono % | 3408 mIoU % |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in ORDER:
        z = m[a]; lines.append(f"| {a} | {res[a]['best_epoch']} | {100*z['miou_4class']:.2f} | {100*z['iou_monolayer']:.2f} | {100*z['iou_fewlayer']:.2f} | "
                               f"{100*z['thin_mean_iou']:.2f} | {100*z['boundary_f1_foreground']:.2f} | {100*z['few_to_mono_rate']:.2f} | {100*s3408(a):.2f} |")
    lines += ['', f"Main screen passed: {out['screening_passed']}", '', '```', json.dumps(checks, indent=1), '```', '',
              'Single training seed 42; val25 used for selection; no test evaluation. Without-m109 values are descriptive only.']
    (G.BATCH/'final_analysis.md').write_text('\n'.join(lines)+'\n')
    return out


def coordinator():
    lock = open(G.ROOT/'outputs/.experiments.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = freeze(); guard(plan); status(stage='starting', pid=os.getpid(), plan_sha256=G.digest(PLAN))
    for a in ORDER:
        run = Path(plan['runs'][a]); st = run/'status.json'
        if st.exists() and json.loads(st.read_text()).get('status') == 'completed': continue
        status(stage=f'training_{a}', started_arm=now())
        with open(G.BATCH/f'train_{a}.log', 'a') as log:
            rc = subprocess.call([PY, '-B', '-W', 'ignore', __file__, '--worker', a], stdout=log, stderr=subprocess.STDOUT)
        if rc: status(stage=f'failed_{a}', returncode=rc); sys.exit(rc)
        status(**{f'completed_{a}': now()})
    for a in ('E2', 'E3'):
        if (G.BATCH/f'secondary_{a}_3408.json').exists(): continue
        status(stage=f'secondary_{a}_3408')
        with open(G.BATCH/f'secondary_{a}.log', 'a') as log:
            rc = subprocess.call([PY, '-B', '-W', 'ignore', __file__, '--secondary', a], stdout=log, stderr=subprocess.STDOUT)
        if rc: status(stage=f'failed_secondary_{a}', returncode=rc); sys.exit(rc)
    out = report(plan); guard(plan)
    status(stage='completed', screening_passed=out['screening_passed'], finished=now())


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker': worker(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == '--secondary': secondary(sys.argv[2])
    else: coordinator()

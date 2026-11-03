"""Explicitly authorized test27 inference for the GLP-Flow study (E0-E3, E3R).

Weights are the frozen validation-best EMA checkpoints; no reselection on test. Historical test exposure applies.
GT enters only the metric computation after prediction (G.evaluate -> G.predict takes RGB/thumbnail/ID only).
"""
import os, sys, json, time, fcntl, shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from PIL import Image
import glpflow_core as G
from conditional_diffusion.data import PALETTE

OUT = G.ROOT/'outputs/inference/glpflow_test_20261022'
RUNS = {'E0': G.RUN_ROOT/'p1_glpflow_20261021_e0', 'E1': G.RUN_ROOT/'p1_glpflow_20261021_e1',
        'E2': G.RUN_ROOT/'p1_glpflow_20261021_e2', 'E3': G.RUN_ROOT/'p1_glpflow_20261021_e3',
        'E3R': G.RUN_ROOT/'p1_glpflow_rollout_20261022_e3r'}
KEYS = ['miou_4class', 'iou_monolayer', 'iou_fewlayer', 'iou_multilayer', 'thin_mean_iou',
        'boundary_f1_foreground', 'few_to_mono_rate']


def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def status(stage, **kw):
    G.atomic_json(OUT/'status.json', dict(stage=stage, pid=os.getpid(), updated_bjt=now(), **kw))


def image_miou(m):
    i = np.diag(m).astype(float); u = m.sum(0)+m.sum(1)-np.diag(m); v = u > 0
    return float((i[v]/u[v]).mean())


def main():
    lock = open(G.ROOT/'outputs/.experiments.lock', 'a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not OUT.exists(), 'preserve the previous export; choose a new directory'
    assert shutil.disk_usage(G.ROOT).free > 10*1024**3
    OUT.mkdir(parents=True)
    ckpt = {a: r/'checkpoints/best.pt' for a, r in RUNS.items()}
    plan = dict(created_bjt=now(), authorization='User explicitly requested test inference (2026-10-22)', split='test',
                historical_test_exposure=True, selected_by='validation-best EMA checkpoint; no test reselection',
                sampling_seed=3407, script_sha256=G.digest(__file__), core_sha256=G.digest(G.ROOT/'work/glpflow_core.py'),
                test_split_sha256=G.digest(G.ROOT/'data_manifests/splits/test.txt'),
                checkpoints={a: dict(path=str(p), sha256=G.digest(p)) for a, p in ckpt.items()})
    G.atomic_json(OUT/'test_plan.json', plan)
    results, rows = {}, []
    for arm, path in ckpt.items():
        status('testing', arm=arm)
        state = torch.load(path, map_location='cpu', weights_only=False); c = state['config']
        assert c['arm'] == arm and c['sampling_seed'] == 3407
        val = json.loads((RUNS[arm]/'best_val_metrics.json').read_text())
        assert val['epoch'] == state['epoch']
        G.configure(c['seed']); model = G.GLPFlowNet(c).cuda(); model.load_state_dict(state['model'], strict=True); del state
        (OUT/arm/'masks').mkdir(parents=True); (OUT/arm/'color').mkdir()
        def save(ident, pred, arm=arm):
            Image.fromarray(pred).save(OUT/arm/'masks'/f'{ident}.png', compress_level=3)
            Image.fromarray(PALETTE[pred]).save(OUT/arm/'color'/f'{ident}.png', compress_level=3)
        metrics, mats, _ = G.evaluate(model, c, split='test', on_image=save)
        assert metrics['image_count'] == 27
        np.savez_compressed(OUT/f'{arm}_confusion.npz', **mats)
        for ident, m in zip(mats['ids'], mats['per_image']):
            rows.append(dict(arm=arm, image_id=str(ident), miou_present=image_miou(m)))
        results[arm] = dict(best_epoch=val['epoch'], test=metrics, val=val['metrics'])
        G.atomic_json(OUT/'test_metrics.json', results)
        print(arm, 'test mIoU %.4f  val %.4f  %.0fs' % (100*metrics['miou_4class'], 100*val['metrics']['miou_4class'], metrics['seconds']), flush=True)
        del model; torch.cuda.empty_cache()
    G.atomic_json(OUT/'per_image_miou.json', rows)
    lines = ['# GLP-Flow test27 results (val-best checkpoints, sampling seed 3407)', '',
             '| Arm | Ep | ' + ' | '.join(KEYS) + ' | sec |', '|---|---:|' + '---:|'*(len(KEYS)+1)]
    for arm, r in results.items():
        lines.append(f"| {arm} | {r['best_epoch']} | " + ' | '.join(f"{100*r['test'][k]:.2f}" for k in KEYS) + f" | {r['test']['seconds']:.0f} |")
    lines += ['', 'Historical test exposure; fixed validation-best weights; single training seed 42; not a blind test.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    status('completed', finished_bjt=now())


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        if OUT.exists(): status('failed', error=repr(e))
        raise

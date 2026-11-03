"""Stage 1 of the Chapter-4 extension: E3R sampling-process export on val25 (forward only, 2026-10-24).

Runs the frozen validation-best E3R EMA checkpoint (ep60, sampling seed 3407, K=4) with exactly the arithmetic of
analyze_glpflow_e3r_20261022.chain_probs and additionally keeps, per image,
  states (5,4,H/4,W/4): source x0, the inputs x1..x3 of steps 2-4 and the final state x4;
  probs  (4,4,H/4,W/4): endpoint class probabilities q_k of each step.
Window results are fused by overlap averaging (the same rule as the probabilities; for states it is a visualization
convention). Arrays are float16 on the [::4, ::4] grid of the existing confidence_div4 maps.
Checks: every step's full-resolution argmax equals the saved trajectories/E3R endpoints, and the last step's per-image
confusion matrices equal best_val_confusion.npz. No test images, no optimizer, GT only enters the checks.
Rerunning resumes: images whose npz exists are re-verified from the saved arrays only if the report lists them.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import json
import time
import fcntl
import shutil
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import glpflow_core as G

RUN = G.ROOT / 'outputs/P1/glpflow/seed_42/p1_glpflow_rollout_20261022_e3r'
TRAJ = G.ROOT / 'outputs/analysis/glpflow_e3r_mechanism_20261022/trajectories/E3R'
OUT = G.ROOT / 'outputs/analysis/e3r_process_export_20261024'
STEPS = 4


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


@torch.no_grad()
def chain_states(model, c, d, steps=STEPS):
    """chain_probs arithmetic (identical accumulation order) plus fused states."""
    image, thumb, ident = d['image'], d['thumb'], d['id']
    h, w = image.shape[-2:]
    sz = c['crop_size']
    hw = ((h // 64) * 64, (w // 64) * 64)
    ctx = model.global_context([thumb.cuda()])
    gen = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'{ident}:{c["sampling_seed"]}'.encode()).digest()[:8], 'little') % (2**63 - 1))
    field = torch.randn((4, h, w), generator=gen)
    totals = [torch.zeros((4, h, w)) for _ in range(steps)]
    states = [torch.zeros((4, h, w)) for _ in range(steps + 1)]
    counts = torch.zeros((h, w))
    coords = [(y, x) for y in G.positions(h, sz, c['stride']) for x in G.positions(w, sz, c['stride'])]
    for s in range(0, len(coords), c['window_batch_size']):
        part = coords[s:s + c['window_batch_size']]
        b = len(part)
        img = torch.stack([image[:, y:y + sz, x:x + sz] for y, x in part]).cuda()
        boxes = torch.tensor([[y / hw[0], x / hw[1], (y + sz) / hw[0], (x + sz) / hw[1]] for y, x in part])
        feats = model.local(img)
        p, umap, gtok, gpos, _ = G.window_conditions(model, c, ctx * b, boxes, sz, 'cuda')
        eps = torch.stack([field[:, y:y + sz, x:x + sz] for y, x in part]).cuda()
        x = G.make_source(c, p, umap, eps)
        sc = torch.zeros_like(p)
        lv = torch.full((b,), G.level_for_steps(steps), device='cuda', dtype=torch.long)
        for st, (y, x_) in zip(x.cpu(), part):
            states[0][:, y:y + sz, x_:x_ + sz] += st
        for k in range(steps):
            t = torch.full((b,), k / steps, device='cuda')
            sc = model.denoise(feats, x, p, umap, sc, t, lv, gtok, gpos, boxes).softmax(1)
            for q, (y, x_) in zip(sc.cpu(), part):
                totals[k][:, y:y + sz, x_:x_ + sz] += q
            x = G.euler(x, 2 * sc - 1, t, torch.full_like(t, 1 / steps))
            for st, (y, x_) in zip(x.cpu(), part):
                states[k + 1][:, y:y + sz, x_:x_ + sz] += st
        for y, x_ in part:
            counts[y:y + sz, x_:x_ + sz] += 1
    assert bool((counts > 0).all())
    return [tot / counts for tot in totals], [st / counts for st in states]


def main():
    if os.environ.get('CH4_COORDINATOR_HOLDS_LOCK') != '1':  # standalone run: take the experiments lock
        lock = open(G.ROOT / 'outputs/.experiments.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert shutil.disk_usage(G.ROOT).free > 10 * 1024**3, 'need >10 GiB free'
    OUT.mkdir(parents=True, exist_ok=True)
    rpath = OUT / 'report.json'
    report = json.loads(rpath.read_text()) if rpath.exists() else {}
    if report.get('all_checks_passed'):
        print('export already completed', flush=True)
        return
    state = torch.load(RUN / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    c = state['config']
    assert c['arm'] == 'E3R' and state['epoch'] == 60 and c['sampling_seed'] == 3407
    G.configure(42)
    model = G.GLPFlowNet(c).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    ref = np.load(RUN / 'best_val_confusion.npz')
    report.update(started=report.get('started', now()), scope='val25 forward-only, E3R best EMA (ep60), seed 3407, K=4',
                  script_sha256=G.digest(__file__), core_sha256=G.digest(G.ROOT / 'work/glpflow_core.py'),
                  checkpoint_sha256=G.digest(RUN / 'checkpoints/best.pt'), grid='[::4, ::4] of 1922x2560 -> 481x640, float16',
                  arrays=dict(states='x0, x1, x2, x3, x4 (input of step k is states[k])', probs='q_k endpoint probabilities, k=0..3'),
                  images=report.get('images', {}))
    G.atomic_json(rpath, report)

    ds = G.GLPDataset(c, 'val', train=False)
    ids = [str(i) for i in ref['ids']]
    assert ds.ids == ids
    for i in range(len(ds)):
        ident = ids[i]
        if report['images'].get(ident, {}).get('passed') and (OUT / f'{ident}.npz').exists():
            continue
        t0 = time.perf_counter()
        d = ds[i]
        probs, states = chain_states(model, c, d)
        preds = [p.argmax(0).numpy().astype(np.uint8) for p in probs]
        saved = np.load(TRAJ / f'{ident}.npz')['endpoints']
        steps_equal = [bool(np.array_equal(pr, saved[k])) for k, pr in enumerate(preds)]
        conf_equal = bool(np.array_equal(G.confusion(d['mask'].numpy(), preds[-1]), ref['per_image'][i]))
        np.savez_compressed(OUT / f'{ident}.npz',
                            states=torch.stack(states)[:, :, ::4, ::4].numpy().astype(np.float16),
                            probs=torch.stack(probs)[:, :, ::4, ::4].numpy().astype(np.float16))
        report['images'][ident] = dict(endpoints_equal_saved=steps_equal, final_confusion_equal_best_val=conf_equal,
                                       passed=all(steps_equal) and conf_equal, seconds=round(time.perf_counter() - t0, 2))
        G.atomic_json(rpath, report)
        print(now(), ident, report['images'][ident], flush=True)
        if not report['images'][ident]['passed']:
            report['all_checks_passed'] = False
            G.atomic_json(rpath, report)
            sys.exit(1)

    # every image's final confusion equals best_val_confusion (checked above), so the global value follows from it
    report['val_miou_4class'] = G.classification_metrics(ref['per_image'].sum(0))['miou_4class']
    report['size_mib'] = round(sum(f.stat().st_size for f in OUT.glob('*.npz')) / 2**20, 1)
    report['all_checks_passed'] = all(v['passed'] for v in report['images'].values()) and len(report['images']) == 25
    report['finished'] = now()
    G.atomic_json(rpath, report)
    (OUT / 'README.md').write_text(
        '# E3R 采样过程导出（val25，仅前向，2026-10-24）\n\n'
        '- 权重：E3R val-best EMA（ep60），采样种子 3407，K=4，与选模设置相同。\n'
        '- `<id>.npz`：`states`（5,4,481,640）为初态 x0、第2–4步输入 x1–x3 与终态 x4；`probs`（4,4,481,640）为每步端点四类概率 q_k。\n'
        '- 网格：整图 1922×2560 的 `[::4, ::4]` 取样，float16；与 `glpflow_e3r_mechanism_20261022/trajectories/E3R` 的 `confidence_div4` 同网格。\n'
        '- 融合：重叠窗口平均；状态的平均融合仅作可视化约定。\n'
        '- 校验见 `report.json`：每步全分辨率 argmax 与已存轨迹逐像素一致，第4步逐图混淆矩阵与 best_val_confusion 一致。\n')
    print('EXPORT DONE', report['all_checks_passed'], report['val_miou_4class'], flush=True)
    if not report['all_checks_passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()

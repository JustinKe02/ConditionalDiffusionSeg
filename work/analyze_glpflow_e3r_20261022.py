"""Forward-only mechanism analysis for Chapter 3 figures (val25; no optimizer steps; no test inference).

Runs on the frozen validation-best EMA checkpoints of E0/E1/E2/E3/E3R:
  1. E0/E1/E2 standard prediction (timed, prediction only) -> val masks, distance-to-boundary accuracy.
  2. E3 and E3R K=4 self chains -> per-step endpoints (metrics, maps), E3R global prior/uncertainty maps.
  3. E3R oracle K=4 chain (GT-interpolated state at k>0, diagnostic only) -> state-gap comparison with E3.
  4. E3R standard prediction with K=1/2/4/8 (timed) -> quality vs steps vs latency.
Test27 distance-to-boundary accuracy is computed on the CPU from the already saved test masks (no new test inference).
GT enters only metrics and the explicitly labelled oracle diagnostic.
"""
import os, sys, json, time, fcntl, shutil, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from PIL import Image
import glpflow_core as G
import diagnose_glpflow_20261022 as D

OUT = G.ROOT/'outputs/analysis/glpflow_e3r_mechanism_20261022'
TEST = G.ROOT/'outputs/inference/glpflow_test_20261022'
RUNS = {'E0': G.RUN_ROOT/'p1_glpflow_20261021_e0', 'E1': G.RUN_ROOT/'p1_glpflow_20261021_e1',
        'E2': G.RUN_ROOT/'p1_glpflow_20261021_e2', 'E3': G.RUN_ROOT/'p1_glpflow_20261021_e3',
        'E3R': G.RUN_ROOT/'p1_glpflow_rollout_20261022_e3r'}
RADII = (2, 4, 8, 16)  # Chebyshev distance strata to the nearest GT class edge: <=2, (2,4], (4,8], (8,16], >16
STRATA = ['0-2', '2-4', '4-8', '8-16', '>16']


def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def load(arm):
    state = torch.load(RUNS[arm]/'checkpoints/best.pt', map_location='cpu', weights_only=False); c = state['config']
    assert c['arm'] == arm
    model = G.GLPFlowNet(c).cuda(); model.load_state_dict(state['model'], strict=True); model.eval()
    return model, c, state['epoch']


def strata(truth):
    lab = np.full(truth.shape, len(RADII), np.uint8)
    for i, r in reversed(list(enumerate(RADII))): lab[D.band(truth, r)] = i
    return lab


class Acc:
    """Global confusion, boundary counts and distance-stratum correctness per GT class."""
    def __init__(self):
        self.m = np.zeros((4, 4), np.int64); self.b = np.zeros((4, 4), np.int64)
        self.ok = np.zeros((len(STRATA), 4), np.int64); self.n = np.zeros((len(STRATA), 4), np.int64); self.per_image = []
    def add(self, truth, pred, lab, tol):
        m = G.confusion(truth, pred); self.m += m; self.per_image.append(m)
        self.b += G.boundary_counts(truth, pred, tol); hit = pred == truth
        for s in range(len(STRATA)):
            for k in range(4):
                sel = (lab == s) & (truth == k); self.n[s, k] += int(sel.sum()); self.ok[s, k] += int(hit[sel].sum())
    def summary(self):
        m = G.classification_metrics(self.m); bd = G.summarize_boundaries(self.b)
        acc = {s: dict(all=float(self.ok[i].sum()/self.n[i].sum()),
                       thin=float(self.ok[i, 1:3].sum()/max(1, self.n[i, 1:3].sum())),
                       pixels=int(self.n[i].sum()), thin_pixels=int(self.n[i, 1:3].sum())) for i, s in enumerate(STRATA)}
        keys = ['miou_4class', 'iou_monolayer', 'iou_fewlayer', 'iou_multilayer', 'thin_mean_iou', 'few_to_mono_rate']
        return dict(**{k: m[k] for k in keys}, boundary_f1_foreground=bd['boundary_f1_foreground'],
                    boundary_f1=dict(mono=bd['boundary_f1_monolayer'], few=bd['boundary_f1_fewlayer'], multi=bd['boundary_f1_multilayer']),
                    distance_accuracy=acc, correct=self.ok.tolist(), total=self.n.tolist())


@torch.no_grad()
def chain_probs(model, c, d, steps):
    """Same arithmetic as D.chain(mode='self') but returns per-step fused probabilities and the global prior."""
    image, thumb, ident = d['image'], d['thumb'], d['id']
    h, w = image.shape[-2:]; sz = c['crop_size']; hw = ((h//64)*64, (w//64)*64)
    ctx = model.global_context([thumb.cuda()])
    gen = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'{ident}:{c["sampling_seed"]}'.encode()).digest()[:8], 'little') % (2**63-1))
    field = torch.randn((4, h, w), generator=gen)
    totals = [torch.zeros((4, h, w)) for _ in range(steps)]; counts = torch.zeros((h, w))
    coords = [(y, x) for y in G.positions(h, sz, c['stride']) for x in G.positions(w, sz, c['stride'])]
    for s in range(0, len(coords), c['window_batch_size']):
        part = coords[s:s+c['window_batch_size']]; b = len(part)
        img = torch.stack([image[:, y:y+sz, x:x+sz] for y, x in part]).cuda()
        boxes = torch.tensor([[y/hw[0], x/hw[1], (y+sz)/hw[0], (x+sz)/hw[1]] for y, x in part])
        feats = model.local(img)
        p, umap, gtok, gpos, _ = G.window_conditions(model, c, ctx*b, boxes, sz, 'cuda')
        eps = torch.stack([field[:, y:y+sz, x:x+sz] for y, x in part]).cuda()
        x = G.make_source(c, p, umap, eps); sc = torch.zeros_like(p)
        lv = torch.full((b,), G.level_for_steps(steps), device='cuda', dtype=torch.long)
        for k in range(steps):
            t = torch.full((b,), k/steps, device='cuda')
            sc = model.denoise(feats, x, p, umap, sc, t, lv, gtok, gpos, boxes).softmax(1)
            for q, (y, x_) in zip(sc.cpu(), part): totals[k][:, y:y+sz, x_:x_+sz] += q
            x = G.euler(x, 2*sc-1, t, torch.full_like(t, 1/steps))
        for y, x_ in part: counts[y:y+sz, x_:x_+sz] += 1
    prior = ctx[0][1].softmax(0).cpu()
    return [tot/counts for tot in totals], prior


def png(path, a): path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(a).save(path, compress_level=3)


def main():
    lock = open(G.ROOT/'outputs/.experiments.lock', 'a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not OUT.exists(), 'preserve previous analysis'
    assert shutil.disk_usage(G.ROOT).free > 10*1024**3
    OUT.mkdir(parents=True); G.configure(42)
    report = dict(started=now(), scope='val25 forward-only on validation-best EMA checkpoints; test27 strata from saved masks only',
                  script_sha256=G.digest(__file__), core_sha256=G.digest(G.ROOT/'work/glpflow_core.py'),
                  checkpoints={a: G.digest(r/'checkpoints/best.pt') for a, r in RUNS.items()},
                  distance='Chebyshev distance to nearest GT class edge (both sides of edge counted as 0)', runs={}, checks={})
    def save(name, value):
        report['runs'][name] = value; G.atomic_json(OUT/'report.json', report)
        print(now(), name, json.dumps({k: v for k, v in value.items() if k in ('miou_4class', 'boundary_f1_foreground', 'seconds_per_image')}), flush=True)
    ds = G.GLPDataset(G.config('E0'), 'val', train=False)
    data = [ds[i] for i in range(len(ds))]; labs = [strata(d['mask'].numpy()) for d in data]
    ids = [d['id'] for d in data]

    # 1. standard predictions, timed (prediction only, excluding data loading and metrics)
    def timed_predict(arm, model, c, steps=None, keep=True):
        acc, secs = Acc(), []; torch.cuda.reset_peak_memory_stats()
        for d, lab in zip(data, labs):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            pred, _ = G.predict(model, c, d['image'], d['thumb'], d['id'], steps)
            torch.cuda.synchronize(); secs.append(time.perf_counter()-t0)
            acc.add(d['mask'].numpy(), pred, lab, c['boundary_tolerance'])
            if keep: png(OUT/'val_predictions'/arm/f"{d['id']}.png", pred)
        return acc, dict(seconds_per_image=float(np.mean(secs)), seconds_per_image_excl_first=float(np.mean(secs[1:])),
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    def matches_record(arm, acc):
        ref = np.load(RUNS[arm]/'best_val_confusion.npz')
        return bool(np.array_equal(np.array(acc.per_image), ref['per_image']) and list(ref['ids']) == ids)

    for arm in ('E0', 'E1', 'E2'):
        model, c, ep = load(arm); acc, tm = timed_predict(arm, model, c)
        report['checks'][f'{arm}_reproduces_best_val'] = matches_record(arm, acc)
        save(arm, dict(epoch=ep, steps=c['sampling_steps'] if c['flow'] else 1, **acc.summary(), **tm)); del model; torch.cuda.empty_cache()

    # 2. per-step K=4 self chains with saved maps (E3, E3R)
    for arm in ('E3', 'E3R'):
        model, c, ep = load(arm); accs = [Acc() for _ in range(4)]; changed = np.zeros(3, np.int64); total_px = 0
        for d, lab in zip(data, labs):
            probs, prior = chain_probs(model, c, d, 4); truth = d['mask'].numpy()
            preds = [p.argmax(0).numpy().astype(np.uint8) for p in probs]
            for a, pr in zip(accs, preds): a.add(truth, pr, lab, c['boundary_tolerance'])
            changed += [int((preds[k+1] != preds[k]).sum()) for k in range(3)]; total_px += truth.size
            conf = [(255*p.max(0).values[::4, ::4]).round().byte().numpy() for p in probs]
            extra = {}
            if arm == 'E3R':
                ent = (-(prior.clamp_min(1e-6)*prior.clamp_min(1e-6).log()).sum(0)/np.log(4)).numpy()
                extra = dict(prior=prior.numpy().astype(np.float16), prior_entropy=ent.astype(np.float16))
            (OUT/'trajectories'/arm).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OUT/'trajectories'/arm/f"{d['id']}.npz", endpoints=np.stack(preds), confidence_div4=np.stack(conf), **extra)
            png(OUT/'val_predictions'/arm/f"{d['id']}.png", preds[-1])
        report['checks'][f'{arm}_k4_final_reproduces_best_val'] = matches_record(arm, accs[-1])
        save(f'{arm}_self_k4', dict(epoch=ep, per_step=[dict(step=k, t=k/4, **a.summary()) for k, a in enumerate(accs)],
                                    changed_pixel_fraction_between_steps=(changed/total_px).tolist(),
                                    miou_4class=accs[-1].summary()['miou_4class'], boundary_f1_foreground=accs[-1].summary()['boundary_f1_foreground']))
        if arm == 'E3R':
            # 3. oracle state diagnostic, identical protocol to the recorded E3 diagnosis
            r = D.run(model, c, 4, 'oracle'); save('E3R_oracle_k4', dict(epoch=ep, **r))
            # 4. quality / latency vs number of steps
            for k in (1, 2, 4, 8):
                acc, tm = timed_predict(arm, model, c, steps=k, keep=False)
                if k == 4: report['checks']['E3R_predict_k4_reproduces_best_val'] = matches_record(arm, acc)
                save(f'E3R_predict_k{k}', dict(epoch=ep, steps=k, **acc.summary(), **tm))
        del model; torch.cuda.empty_cache()

    # 5. test27 distance strata from saved masks (CPU only)
    man = G.read_manifest(G.config('E0')['manifest']); root = Path(man['dataset_root'])
    tids = Path(man['splits']['test']['snapshot_path']).read_text().split(); tacc = {a: Acc() for a in RUNS}
    for ident in tids:
        with Image.open(root/man['mask_pattern'].format(id=ident)) as im: truth = np.array(im)
        lab = strata(truth)
        for a in RUNS:
            with Image.open(TEST/a/'masks'/f'{ident}.png') as im: tacc[a].add(truth, np.array(im), lab, 2)
    for a in RUNS:
        ref = np.load(TEST/f'{a}_confusion.npz')
        report['checks'][f'test_{a}_masks_match_confusion'] = bool(np.array_equal(np.array(tacc[a].per_image), ref['per_image']))
        save(f'test_{a}', tacc[a].summary())
    report['finished'] = now(); report['all_checks_passed'] = all(report['checks'].values())
    G.atomic_json(OUT/'report.json', report); print('DONE', json.dumps(report['checks']), flush=True)


if __name__ == '__main__':
    main()

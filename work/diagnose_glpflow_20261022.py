"""Forward-only diagnosis of the completed GLP-Flow batch (no optimizer steps, val25 only, no test).

Questions: does multi-step flow sampling help (1/2/4/8 steps)? Does the model rely on ground truth
present in the GT-interpolated training state (oracle state vs self-generated state at equal t)?
Ground truth enters only the explicitly labelled oracle diagnostic, never the normal predictions.
"""
import sys, json, time, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.nn import functional as F
import glpflow_core as G

OUT = G.ROOT/'outputs/analysis/glpflow_diagnosis_20261022'
OLD = G.BATCH
# Pre-registered before any diagnostic forward was run.
RULE = dict(
    confirmed_if_all=dict(
        oracle_minus_self_miou_at_t025_pp_min=3.0,
        self_final_k4_boundary_below_E1_pp_min=5.0),
    action_if_confirmed='train one sampler-aligned E3R (rollout states on the K=4 grid), compare with E1/E3, report and stop',
    action_if_not='stop; recommend Chapter 3 contribution = global-local architecture (E1), diffusion only as control')


def load(arm):
    plan = json.loads((OLD/'plan.json').read_text()); c = plan['configs'][arm]
    state = torch.load(Path(plan['runs'][arm])/'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model = G.GLPFlowNet(c).cuda(); model.load_state_dict(state['model']); model.eval()
    return model, c, state['epoch']


def band(mask, radius=8):
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    edge = F.max_pool2d(F.pad(m, (1, 1, 1, 1), mode='replicate'), 3, 1) != -F.max_pool2d(-F.pad(m, (1, 1, 1, 1), mode='replicate'), 3, 1)
    return (F.max_pool2d(edge.float(), 2*radius+1, 1, radius) > 0)[0, 0].numpy()


@torch.no_grad()
def chain(model, c, d, steps, mode='self', seed=3407):
    """Per-step full-image endpoint predictions. mode='oracle' replaces the state at k>0 by the GT interpolation."""
    image, thumb, ident = d['image'], d['thumb'], d['id']
    h, w = image.shape[-2:]; sz = c['crop_size']; hw = ((h//64)*64, (w//64)*64)
    ctx = model.global_context([thumb.cuda()])
    gen = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f'{ident}:{seed}'.encode()).digest()[:8], 'little') % (2**63-1))
    field = torch.randn((4, h, w), generator=gen) if c['flow'] else None
    y_full = G.encode_targets(d['mask'][None])[0] if mode == 'oracle' else None
    n = steps if c['flow'] else 1
    totals = [torch.zeros((4, h, w)) for _ in range(n)]; counts = torch.zeros((h, w))
    coords = [(y, x) for y in G.positions(h, sz, c['stride']) for x in G.positions(w, sz, c['stride'])]
    for s in range(0, len(coords), c['window_batch_size']):
        part = coords[s:s+c['window_batch_size']]; b = len(part)
        img = torch.stack([image[:, y:y+sz, x:x+sz] for y, x in part]).cuda()
        boxes = torch.tensor([[y/hw[0], x/hw[1], (y+sz)/hw[0], (x+sz)/hw[1]] for y, x in part])
        if not c['flow']:
            logits, _ = G.infer_windows(model, c, img, boxes, ctx*b, None, 1); probs = [logits.softmax(1).cpu()]
        else:
            feats = model.local(img)
            p, umap, gtok, gpos, _ = G.window_conditions(model, c, ctx*b, boxes, sz, 'cuda')
            eps = torch.stack([field[:, y:y+sz, x:x+sz] for y, x in part]).cuda()
            x0 = G.make_source(c, p, umap, eps); x = x0; sc = torch.zeros_like(p)
            yw = None if y_full is None else torch.stack([y_full[:, y:y+sz, x_:x_+sz] for y, x_ in part]).cuda()
            lv = torch.full((b,), G.level_for_steps(steps), device='cuda', dtype=torch.long); probs = []
            for k in range(steps):
                t = torch.full((b,), k/steps, device='cuda')
                if mode == 'oracle' and k > 0: x = (1-k/steps)*x0+(k/steps)*yw
                logits = model.denoise(feats, x, p, umap, sc, t, lv, gtok, gpos, boxes)
                sc = logits.softmax(1); probs.append(sc.cpu()); x = G.euler(x, 2*sc-1, t, torch.full_like(t, 1/steps))
        for k, pr in enumerate(probs):
            for q, (y, x_) in zip(pr, part): totals[k][:, y:y+sz, x_:x_+sz] += q
        for y, x_ in part: counts[y:y+sz, x_:x_+sz] += 1
    return [(tot/counts).argmax(0).numpy().astype(np.uint8) for tot in totals]


def run(model, c, steps, mode='self'):
    ds = G.GLPDataset(c, 'val', train=False); n = steps if c['flow'] else 1
    acc = [dict(m=np.zeros((4, 4), np.int64), b=np.zeros((4, 4), np.int64), band=[0, 0], off=[0, 0]) for _ in range(n)]
    t0 = time.perf_counter()
    for i in range(len(ds)):
        d = ds[i]; truth = d['mask'].numpy(); bd = band(truth)
        for a, pred in zip(acc, chain(model, c, d, steps, mode)):
            a['m'] += G.confusion(truth, pred); a['b'] += G.boundary_counts(truth, pred, c['boundary_tolerance'])
            ok = pred == truth
            a['band'][0] += int(ok[bd].sum()); a['band'][1] += int(bd.sum())
            a['off'][0] += int(ok[~bd].sum()); a['off'][1] += int((~bd).sum())
    out = []
    for k, a in enumerate(acc):
        m = G.classification_metrics(a['m']); bnd = G.summarize_boundaries(a['b'])
        out.append(dict(step=k, t=k/steps if c['flow'] else 0., miou=m['miou_4class'], thin_mean=m['thin_mean_iou'],
                        mono=m['iou_monolayer'], few=m['iou_fewlayer'], boundary_f1_fg=bnd['boundary_f1_foreground'],
                        band8_accuracy=a['band'][0]/a['band'][1], off_band_accuracy=a['off'][0]/a['off'][1]))
    return dict(mode=mode, steps=steps, seconds=time.perf_counter()-t0, per_step=out)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); G.configure(42)
    report = dict(rule=RULE, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), runs={})
    G.atomic_json(OUT/'diagnosis.json', report)
    def save(name, value):
        report['runs'][name] = value; G.atomic_json(OUT/'diagnosis.json', report)
        print(name, json.dumps(value['per_step'][-1] if 'per_step' in value else value)[:400], flush=True)
    m1, c1, e1 = load('E1'); save('E1', dict(epoch=e1, **run(m1, c1, 1))); del m1
    m3, c3, e3 = load('E3')
    save('E3_self_k4', dict(epoch=e3, **run(m3, c3, 4, 'self')))
    save('E3_oracle_k4', dict(epoch=e3, **run(m3, c3, 4, 'oracle')))
    for k in (1, 2, 8): save(f'E3_self_k{k}', dict(epoch=e3, **run(m3, c3, k, 'self')))
    recorded = json.loads((OLD/'comparison_summary.json').read_text())['results']['E3']['metrics']['miou_4class']
    s, o, ref = report['runs']['E3_self_k4']['per_step'], report['runs']['E3_oracle_k4']['per_step'], report['runs']['E1']['per_step'][0]
    gap = 100*(o[1]['miou']-s[1]['miou']); bgap = 100*(ref['boundary_f1_fg']-s[-1]['boundary_f1_fg'])
    report['decision'] = dict(
        reproduces_recorded_E3_k4=abs(s[-1]['miou']-recorded) < 1e-9,
        oracle_minus_self_miou_at_t025_pp=gap, self_final_boundary_below_E1_pp=bgap,
        multi_step_gain_pp=100*(s[-1]['miou']-s[0]['miou']),
        confirmed=bool(gap >= RULE['confirmed_if_all']['oracle_minus_self_miou_at_t025_pp_min']
                       and bgap >= RULE['confirmed_if_all']['self_final_k4_boundary_below_E1_pp_min']))
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z'); G.atomic_json(OUT/'diagnosis.json', report)
    print('DECISION', json.dumps(report['decision']), flush=True)

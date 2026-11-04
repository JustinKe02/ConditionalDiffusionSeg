"""Matched Direct calibration heads. Only reference location changes between arms.

Descriptors use image data only. Training labels supervise the conditional
mono/few decision in both arms; no oracle reference, relation label or
auxiliary loss is supplied to either. All coordinates refer to native pixels.
"""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FEATURES = ('rgb', 'g', 'local', 'p')
WIDTH = 19
STEPS = 600
BATCH_SIZE = 512
CAP = 6.0


def _reference_grid(im):
    """Cache the same 32x32 box average for both arms on a 16-pixel lattice."""
    cache = im.get('_layerpair_reference_grid')
    if cache is not None:
        return cache
    h, w = im['p'].shape[:2]
    gh, gw = h // 16, w // 16
    if min(gh, gw) < 2:
        raise ValueError('Expected native images at least 32x32')
    parts = []
    for name in FEATURES:
        arr = np.asarray(im[name])
        blocks = arr[:gh*16, :gw*16].reshape(gh, 16, gw, 16, arr.shape[-1])
        mean16 = blocks.mean(axis=(1, 3), dtype=np.float32)
        mean32 = .25 * (mean16[:-1, :-1] + mean16[1:, :-1]
                        + mean16[:-1, 1:] + mean16[1:, 1:])
        parts.append(mean32)
    grid = np.concatenate(parts, axis=-1)
    if grid.shape[2] != WIDTH:
        raise ValueError(f'Expected RGB(3), prior(4), local(8), probabilities(4); got {grid.shape}')
    anchors = []
    for a in sorted(im['anchors'], key=lambda a: -float(a['score'])):
        x, y = int(a['x']), int(a['y'])
        if float(a['score']) < .85:
            continue
        if x % 16 or y % 16 or x < 0 or y < 0 or x+32 > w or y+32 > h:
            raise ValueError('Reference patch must lie on a valid 16-pixel lattice location')
        # Centre of the 32x32 pixel square, in the sample-index convention.
        component = int(a['component'])
        if component <= 0:
            raise ValueError('Thin anchors must belong to positive prior component IDs')
        anchors.append((y+15.5, x+15.5, component, grid[y//16, x//16]))
    cache = (grid, anchors)
    im['_layerpair_reference_grid'] = cache
    return cache


def descriptors(im, ys, xs):
    """Common pixel, local reference A, same-field reference B, shared gate."""
    ys, xs = np.asarray(ys, dtype=np.intp), np.asarray(xs, dtype=np.intp)
    if ys.ndim != 1 or ys.shape != xs.shape:
        raise ValueError('Use matching one-dimensional native y and x arrays')
    grid, anchors = _reference_grid(im)
    common = np.concatenate([np.asarray(im[name])[ys, xs] for name in FEATURES], axis=-1)
    common = common.astype(np.float32, copy=False)
    # Select the grid patch whose centre is closest to the query, including edges.
    gy = np.clip((ys-8)//16, 0, grid.shape[0]-1)
    gx = np.clip((xs-8)//16, 0, grid.shape[1]-1)
    selfref = grid[gy, gx]
    sameref = np.zeros_like(selfref)
    remaining = common[:, -3] + common[:, -2] >= .20  # p1 + p2
    eligible = np.zeros(len(ys), dtype=bool)
    prior_components = np.asarray(im['prior_components'])
    query_component = prior_components[np.minimum(ys//16,prior_components.shape[0]-1),
                                       np.minimum(xs//16,prior_components.shape[1]-1)]
    for cy, cx, component, anchor in anchors:
        idx = np.flatnonzero(remaining)
        if not len(idx):
            break
        distinct = query_component[idx] != component
        far = idx[distinct & ((ys[idx]-cy)**2 + (xs[idx]-cx)**2 >= 64**2)]
        sameref[far] = anchor
        eligible[far] = True
        remaining[far] = False
    return common, selfref, sameref, eligible


class Calibrator(nn.Module):
    """Bounded additive correction to log[P(few)/P(mono)]."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3*WIDTH, 32), nn.LayerNorm(32), nn.SiLU(),
            nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, q, ref):
        delta = q-ref
        raw = self.mlp(torch.cat((q, delta, delta.abs()), dim=1)).squeeze(1)
        return CAP*torch.tanh(raw/CAP)


def _base_log_odds(p):
    return p[:, 2].clamp_min(1e-12).log() - p[:, 1].clamp_min(1e-12).log()


def train_heads(data, seed=42):
    """Train two independent CPU heads from the same init with identical draws."""
    y = np.asarray(data['y']).ravel()
    active = np.asarray(data['eligible'], dtype=bool).ravel() & np.isin(y, (1, 2))
    idx = np.flatnonzero(active)
    if set(y[idx].tolist()) != {1, 2}:
        raise ValueError('Active training pixels must include mono and few')
    # The caller controls the common per-image/pixel sampling before this step.
    positive_target = torch.from_numpy((y[idx] == 2).astype(np.float32))
    q, a, b, p = [torch.from_numpy(np.ascontiguousarray(np.asarray(data[key])[idx],dtype=np.float32))
                  for key in ('common','selfref','sameref','p')]
    base = _base_log_odds(p)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head_a = Calibrator()
    heads = (head_a, deepcopy(head_a))
    optimizers = [torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001) for head in heads]
    gen = torch.Generator(device='cpu').manual_seed(seed)
    for _ in range(STEPS):
        ids = torch.randint(len(idx),(BATCH_SIZE,),generator=gen)
        for head, opt, ref in zip(heads,optimizers,(a,b)):
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(base[ids] + head(q[ids],ref[ids]),positive_target[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite calibrator training loss')
            loss.backward()
            opt.step()
    return tuple(head.eval() for head in heads)


@torch.inference_mode()
def predict_heads(heads, common, selfref, sameref, p, eligible):
    """Yield two native class-ID arrays, returning Direct exactly where inactive."""
    p = np.asarray(p,dtype=np.float32)
    base = p.argmax(axis=1).astype(np.uint8)
    active = np.flatnonzero(eligible)
    results = []
    for head, ref in zip(heads,(selfref,sameref)):
        pred = base.copy()
        for i in range(0,len(active),16384):
            ids = active[i:i+16384]
            probs = torch.from_numpy(p[ids].copy())
            corr = head(torch.from_numpy(common[ids]),torch.from_numpy(ref[ids]))
            mass = probs[:,1] + probs[:,2]
            q = torch.sigmoid(_base_log_odds(probs)+corr)
            probs[:,1] = mass*(1-q)
            probs[:,2] = mass*q
            new = probs.argmax(1).numpy().astype(np.uint8)
            # Retain exact baseline tie handling even at a zero update.
            delta = corr.numpy() != 0
            pred[ids[delta]] = new[delta]
        results.append(pred)
    return tuple(results)

"""Native-pixel, paired field-level assessment for the frozen-Direct A/B prototype."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

ARMS = ('direct', 'A', 'B')
CLASSES = ('background', 'monolayer', 'fewlayer', 'multilayer')


def _mask(array):
    arr = np.asarray(array)
    if arr.ndim != 2 or not arr.size or arr.dtype.kind not in 'iu' or arr.min() < 0 or arr.max() > 3:
        raise ValueError('Expected one nonempty 2D integer mask with labels 0..3')
    return arr.astype(np.uint8, copy=False)


def _boundary(mask, cls):
    binary = mask == cls
    # Same inner boundary and edge replication as metrics_impl.boundary_counts.
    edge = binary & ~ndimage.minimum_filter(binary, size=3, mode='nearest')
    near = ndimage.maximum_filter(edge, size=5, mode='constant', cval=False)
    return edge, near


def add_image(rows, ident, gt, pred_direct, pred_A, pred_B):
    """Append JSON-compatible counts; no per-pixel evidence is retained."""
    ident = str(ident)
    if any(row['id'] == ident for row in rows):
        raise ValueError(f'Duplicate field ID: {ident}')
    gt = _mask(gt)
    preds = [_mask(x) for x in (pred_direct, pred_A, pred_B)]
    item = {'id': ident}
    labels = gt * 4
    for arm, pred in zip(ARMS, preds):
        if pred.shape != gt.shape:
            raise ValueError(f'{arm}: shape differs from native target')
        matrix = np.bincount((labels + pred).ravel(), minlength=16).reshape(4, 4)
        item[arm] = {'confusion': matrix.tolist(), 'boundaries': []}
    for cls in range(4):
        true_edge, true_near = _boundary(gt, cls)
        n_true = np.count_nonzero(true_edge)
        for arm, pred in zip(ARMS, preds):
            pred_edge, pred_near = _boundary(pred, cls)
            counts = [np.count_nonzero(pred_edge & true_near), np.count_nonzero(pred_edge),
                      np.count_nonzero(true_edge & pred_near), n_true]
            item[arm]['boundaries'].append([int(x) for x in counts])
    rows.append(item)


def _ratio(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)


def _avg(a):
    return _ratio(np.nansum(a, axis=-1), np.isfinite(a).sum(axis=-1))


def _metrics(m, b):
    true, pred = m.sum(axis=-1), m.sum(axis=-2)
    tp = np.diagonal(m, axis1=-2, axis2=-1)
    iou = _ratio(tp, true + pred - tp)
    recall = _ratio(tp, true)
    precision_b = np.nan_to_num(_ratio(b[..., 0], b[..., 1]))
    recall_b = np.nan_to_num(_ratio(b[..., 2], b[..., 3]))
    bf = np.nan_to_num(_ratio(2 * precision_b * recall_b, precision_b + recall_b))
    bf = np.where((b[..., 1] + b[..., 3]) > 0, bf, np.nan)
    result = {
        'few_to_mono_given_few': _ratio(m[..., 2, 1], true[..., 2]),
        'mono_to_few_given_mono': _ratio(m[..., 1, 2], true[..., 1]),
        'recall_monolayer': recall[..., 1], 'recall_fewlayer': recall[..., 2],
        'thin_mean_recall': _avg(recall[..., 1:3]),
        'miou_4class': _avg(iou), 'miou_foreground': _avg(iou[..., 1:]),
        'thin_mean_iou': _avg(iou[..., 1:3]),
        'boundary_f1_foreground': _avg(bf[..., 1:]),
        'monolayer_fewlayer_error_rate': _ratio(m[..., 1, 2] + m[..., 2, 1], true.sum(axis=-1)),
    }
    for k, name in enumerate(CLASSES):
        result['iou_' + name] = iou[..., k]
        result['boundary_f1_' + name] = bf[..., k]
    return result


def _number(v):
    """Undefined class-conditional metrics are explicitly null, never invented zeros."""
    return float(v) if np.isfinite(v) else None


def _counts_and_metrics(m, b):
    return {
        'confusion': m.tolist(), 'boundaries': b.tolist(),
        'true_few_pixels': int(m[2].sum()), 'true_mono_pixels': int(m[1].sum()),
        'few_to_mono_pixels': int(m[2, 1]), 'mono_to_few_pixels': int(m[1, 2]),
        'metrics': {k: _number(v) for k, v in _metrics(m, b).items()},
    }


def _paired(m, b, rng, draws=10000):
    n = len(m)
    weights = rng.multinomial(n, np.full(n, 1/n), size=draws)
    # Each draw resamples entire matched fields and then pools integer counts.
    def pooled(x):
        return (weights @ x.reshape(n, -1)).reshape(draws, *x.shape[1:])
    bs = _metrics(pooled(m), pooled(b))
    point = _metrics(m.sum(axis=0), b.sum(axis=0))
    result = {}
    for target, ref in ((2, 1), (2, 0), (1, 0)):
        delta = {}
        for key, value in bs.items():
            diff = 100 * (value[:, target] - value[:, ref])
            ok = diff[np.isfinite(diff)]
            delta[key] = {
                'estimate_pp': _number(100 * (point[key][target] - point[key][ref])),
                'percentile95_pp': np.percentile(ok, [2.5, 97.5]).tolist() if len(ok) else None,
                'defined_draws': len(ok),
            }
        result[f'{ARMS[target]}_minus_{ARMS[ref]}'] = delta
    return result


def finalize(rows, outdir):
    """Write assessment.json and assessment_per_image.csv, and return the JSON dict."""
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty, distinct paired fields')
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ids = [r['id'] for r in rows]
    m, b = (np.asarray([[r[a][key] for a in ARMS] for r in rows], dtype=np.int64)
            for key in ('confusion', 'boundaries'))
    if m.shape != (len(ids), 3, 4, 4) or b.shape != m.shape:
        raise ValueError('Expected matched 4x4 count tables for all arms')
    if (m < 0).any() or (b < 0).any() or not np.all(m.sum(-1) == m.sum(-1)[:, :1]):
        raise ValueError('Invalid or unpaired native target counts')
    if np.any(b[..., 0] > b[..., 1]) or np.any(b[..., 2] > b[..., 3]) or not np.all(b[..., 3] == b[:, :1, :, 3]):
        raise ValueError('Invalid boundary matching counts')
    totals_m = m.sum(0)
    totals_b = b.sum(0)
    report = dict(
        n_images=len(ids), primary='few_to_mono_given_few = confusion[2,1] / sum(confusion[2,:]); lower is better',
        units='Metrics are fractions. Paired contrasts are percentage points; all deltas are signed.',
        boundary='Three foreground classes; inner 3x3 erosion with replicated image edges, Chebyshev tolerance 2 native pixels.',
        bootstrap='10000 matched image resamples, seed 3407, pooled confusion/boundary counts, percentile intervals. Not training-seed or specimen uncertainty.',
        absent='Undefined field-wise ratios are null and excluded only from that metric’s image mean.',
        pooled={a: _counts_and_metrics(totals_m[j], totals_b[j]) for j, a in enumerate(ARMS)},
        paired=_paired(m, b, np.random.default_rng(3407)),
    )
    per = []
    metrics = _metrics(m, b)
    for i, ident in enumerate(ids):
        for j, arm in enumerate(ARMS):
            r = _counts_and_metrics(m[i,j], b[i,j])
            per.append(dict(id=ident, arm=arm,
                            **{k:v for k,v in r.items() if k not in ('confusion','boundaries','metrics')},
                            **r['metrics']))
    report['image_macro'] = {
        a: {k: {'mean': _number(_avg(v[:, j])), 'n_defined': int(np.isfinite(v[:,j]).sum())}
            for k, v in metrics.items()} for j, a in enumerate(ARMS)
    }
    report['m109'] = None
    if 'm109' in ids and len(ids) > 1:
        i = ids.index('m109')
        focus = {}
        for j, a in enumerate(ARMS):
            fm, fb = m[i,j], b[i,j]
            focus[a] = {
                'm109': _counts_and_metrics(fm, fb),
                'm109_share_of_all_few_to_mono_errors': _number(_ratio(fm[2,1], totals_m[j,2,1])),
                'leave_m109_out': _counts_and_metrics(totals_m[j]-fm, totals_b[j]-fb),
            }
        report['m109'] = focus
    report['per_image_counts'] = rows
    # Each file is complete before replacement, including on retried runs.
    tmp = outdir / 'assessment_per_image.csv.tmp'
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(per[0]))
        writer.writeheader()
        writer.writerows(per)
    tmp.replace(outdir / 'assessment_per_image.csv')
    tmp = outdir / 'assessment.json.tmp'
    tmp.write_text(json.dumps(report, allow_nan=False, indent=2), encoding='utf-8')
    tmp.replace(outdir / 'assessment.json')
    return report

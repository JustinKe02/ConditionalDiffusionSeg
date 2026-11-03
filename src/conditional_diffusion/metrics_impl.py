import numpy as np
import torch
from torch.nn import functional as F
from .data import CLASSES


def confusion(target, prediction):
    return np.bincount((target.astype(np.int64)*4+prediction).ravel(), minlength=16).reshape(4, 4)


def summarize(matrix):
    true, pred, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    union = true+pred-tp
    iou = np.divide(tp, union, out=np.full(4, np.nan), where=union>0)
    f1 = np.divide(2*tp, true+pred, out=np.full(4, np.nan), where=true+pred>0)
    result = {'miou_4class': float(np.nanmean(iou)), 'miou_foreground': float(np.nanmean(iou[1:]))}
    for i, name in enumerate(CLASSES):
        result['iou_'+name] = float(iou[i])
        result['f1_'+name] = float(f1[i])
    return result


def boundary_counts(target, prediction, tolerance=2):
    """Per-class inner 3x3 boundary, Chebyshev matching tolerance in native pixels."""
    result = np.zeros((4, 4), dtype=np.int64)
    for c in range(4):
        borders = []
        for mask in [target, prediction]:
            x = torch.from_numpy((mask == c).astype(np.float32))[None, None]
            erosion = -F.max_pool2d(-F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, 1)
            borders.append((x-erosion)>0)
        gt, pr = borders
        k = 2*tolerance+1
        gt_near = F.max_pool2d(gt.float(), k, 1, tolerance)>0
        pr_near = F.max_pool2d(pr.float(), k, 1, tolerance)>0
        result[c] = [int((pr & gt_near).sum()), int(pr.sum()), int((gt & pr_near).sum()), int(gt.sum())]
    return result


def summarize_boundaries(counts):
    result, foreground = {}, []
    for c, (matched_p, n_p, matched_t, n_t) in enumerate(counts):
        if n_p == 0 and n_t == 0:
            value = float('nan')
        elif n_p == 0 or n_t == 0:
            value = 0.
        else:
            precision, recall = matched_p/n_p, matched_t/n_t
            value = 2*precision*recall/(precision+recall) if precision+recall else 0.
        result['boundary_f1_'+CLASSES[c]] = value
        if c > 0:
            foreground.append(value)
    result['boundary_f1_foreground'] = float(np.nanmean(foreground)) if np.isfinite(foreground).any() else float('nan')
    return result

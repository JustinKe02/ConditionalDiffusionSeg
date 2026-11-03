import torch
from torch.nn import functional as F


def boundary_band(mask):
    x = F.pad(mask[:, None].float(), (1, 1, 1, 1), mode='replicate')
    return (F.max_pool2d(x, 3, 1) != -F.max_pool2d(-x, 3, 1)).squeeze(1)


def losses(logits, mask, config, x0=None, class_weights=None):
    prediction_x0 = logits['x0'] if isinstance(logits, dict) else logits
    logits = logits['logits'] if isinstance(logits, dict) else logits
    ce_map = F.cross_entropy(logits, mask, weight=class_weights, reduction='none')
    # Fixed pixel mean; weighted CE is not normalized again by batch class composition.
    ce = ce_map.mean()
    probs = logits.softmax(1)
    gt = F.one_hot(mask, 4).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    dice = 1-((2*(probs*gt).sum(dims)+1e-6)/(probs.sum(dims)+gt.sum(dims)+1e-6))[1:].mean()
    terms = {'loss_ce': ce, 'loss_dice_fg': dice}
    total = .5*ce+.5*dice
    if x0 is not None:
        terms['loss_x0_mse'] = F.mse_loss(prediction_x0, x0)
        total = total+terms['loss_x0_mse']
    if config['boundary']:
        band = boundary_band(mask)
        terms['loss_boundary'] = ce_map[band].mean() if band.any() else logits.sum()*0
        total = total+config['boundary_weight']*terms['loss_boundary']
    terms['loss_total'] = total
    return total, terms

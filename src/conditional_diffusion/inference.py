import hashlib
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import MEAN, STD, PALETTE
from .metrics_impl import confusion, summarize, boundary_counts, summarize_boundaries


def positions(length, window, stride):
    if not 0 < stride <= window:
        raise ValueError('Stride must be in (0, window]')
    last = max(0, length-window)
    return sorted(set(list(range(0, last+1, stride))+[last]))


def image_seed(ident, seed):
    return int.from_bytes(hashlib.sha256(f'{ident}:{seed}'.encode()).digest()[:8], 'little') % (2**63-1)


@torch.no_grad()
def predict_image(model, diffusion, image, ident, config):
    """Only RGB/image identity/config enter prediction. No ground-truth argument."""
    device = next(model.parameters()).device
    h, w = image.shape[-2:]
    window, stride = config['crop_size'], config['stride']
    pad_h, pad_w = max(h, window), max(w, window)
    image = F.pad(image[None], (0, pad_w-w, 0, pad_h-h), mode='replicate')[0]
    total = torch.zeros((4, pad_h, pad_w))
    counts = torch.zeros((pad_h, pad_w))
    repeats = config['samples_per_image'] if model.method == 'diffusion' else 1
    nfe = 0
    for repeat in range(repeats):
        field = None
        if model.method == 'diffusion':
            generator = torch.Generator().manual_seed(image_seed(ident, config['sampling_seed']+repeat))
            field = torch.randn((4, pad_h, pad_w), generator=generator)
        for y in positions(pad_h, window, stride):
            for x in positions(pad_w, window, stride):
                patch = image[:, y:y+window, x:x+window][None].to(device)
                if field is not None:
                    noise = field[:, y:y+window, x:x+window][None].to(device)
                    logits = diffusion.sample(model, patch, noise, config['sampling_steps'])
                    nfe += config['sampling_steps']
                else:
                    logits = model(patch)
                    nfe += model.iterations if model.method == 'iterative' else 1
                total[:, y:y+window, x:x+window] += logits.softmax(1)[0].cpu()
                counts[y:y+window, x:x+window] += 1
    if not (counts > 0).all():
        raise RuntimeError('Uncovered image pixels')
    probabilities = total[:, :h, :w]/counts[:h, :w]
    return probabilities.argmax(0).numpy().astype(np.uint8), nfe


def preview(image, target, prediction):
    rgb = ((image[:3]*STD+MEAN).clamp(0, 1)*255).byte().permute(1, 2, 0).numpy()
    error = np.zeros_like(rgb)
    error[target != prediction] = [255, 0, 0]
    return {'rgb': rgb, 'ground_truth': PALETTE[target], 'prediction': PALETTE[prediction], 'error': error}


@torch.no_grad()
def evaluate(model, diffusion, dataset, config, on_image=None):
    model.eval()
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    matrices, boundaries, ids, previews = [], np.zeros((4, 4), dtype=np.int64), [], {}
    nfe = 0
    for i in range(len(dataset)):
        image, target, ident = dataset[i]
        prediction, calls = predict_image(model, diffusion, image, ident, config)
        target = target.numpy()
        matrices.append(confusion(target, prediction))
        ids.append(ident)
        nfe += calls
        if config['boundary_metrics']:
            boundaries += boundary_counts(target, prediction, config['boundary_tolerance'])
        if i < config['preview_count']:
            previews[ident] = preview(image, target, prediction)
        if on_image:
            on_image(ident, prediction)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    matrix = np.sum(matrices, axis=0)
    metrics = summarize(matrix)
    if config['boundary_metrics']:
        metrics.update(summarize_boundaries(boundaries))
    metrics['seconds'] = time.perf_counter()-start
    metrics['nfe_total'] = nfe
    metrics['image_count'] = len(ids)
    if device.type == 'cuda':
        metrics['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device)/1024**2
        metrics['peak_reserved_mib'] = torch.cuda.max_memory_reserved(device)/1024**2
    return metrics, {'ids': np.array(ids), 'per_image': np.array(matrices), 'global_matrix': matrix, 'boundary_counts': boundaries}, previews

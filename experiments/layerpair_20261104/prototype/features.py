"""Frozen Direct features. All maps are native-sized HWC float32 NumPy arrays.

p, g and local are means over the same windows in the same order as Direct.
g is the four-class, aligned window prior actually used by the model. local is
channels 0:8 at the input to the segmentation head (before head GroupNorm).
The anchor detector uses the unupsampled whole-field prior, not the final output.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage


def _reference_regions(prior, height, width):
    # Keep final partial cells for query -> component indexing, but demand
    # full 32-square native blocks for reference selection. Cell side is 16.
    prior = prior[:, :math.ceil(height / 16), :math.ceil(width / 16)]
    labels = prior.argmax(axis=0)
    identity = np.zeros(labels.shape, dtype=np.int32)
    anchors = []
    offset = 0
    for label in (1, 2):
        components, n = ndimage.label(labels == label, structure=np.ones((3, 3)))
        present = components > 0
        identity[present] = offset + components[present]
        offset += n
        interior = prior[label, :height // 16, :width // 16]
        if min(interior.shape) < 2:
            continue
        blocks = np.lib.stride_tricks.sliding_window_view(interior, (2, 2))
        confidence = blocks.min(axis=(-2, -1))
        y, x = np.where(confidence >= 0.85)
        order = np.argsort(-confidence[y, x], kind='stable')
        seen = set()
        for j in order:
            row, col = y[j], x[j]
            component = int(identity[row, col])
            if component in seen:
                continue
            seen.add(component)
            anchors.append(dict(x=int(16 * col), y=int(16 * row), label=label,
                                score=float(confidence[row, col]), component=component))
            if len(seen) == 4:
                break
    return anchors, identity


class FrozenDirect:
    def __init__(self, repo, checkpoint, config, device):
        repo = Path(repo).resolve()
        sys.path.insert(0, str(repo / 'src'))
        from conditional_diffusion.data import MaterialDataset
        from conditional_diffusion.diffusion_seg_experiment import build, configure, normalize
        from conditional_diffusion.inference import positions

        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        registered = state['config']
        if config is None:
            config = registered
        if not isinstance(config, dict):
            config = json.loads(Path(config).read_text(encoding='utf-8'))
        # Device and CPU pool size are runtime choices. Validate all scientific settings.
        nonruntime = lambda c: {k: v for k, v in c.items() if k not in ('device', 'cpu_threads')}
        if config['method'] != 'direct' or nonruntime(config) != nonruntime(registered):
            raise ValueError('Expected the matching frozen Direct configuration')
        self.device = torch.device(device)
        self.config = dict(config, device=str(self.device), cpu_threads=min(4, int(config['cpu_threads'])))
        self.epoch = state['epoch']
        self.global_step = state['global_step']
        configure(self.config)
        model, diffusion = build(self.config)
        model.load_state_dict(state['model'], strict=True)
        self.model = model.eval().requires_grad_(False)
        self.state_channels = diffusion.codec.channels
        self._dataset = MaterialDataset
        self._normalize = normalize
        self._positions = positions

    def source(self, split):
        if split not in ('train', 'val'):
            raise ValueError('Prototype only admits train and val')
        return self._dataset(self.config['manifest'], split, augment=False)

    @torch.inference_mode()
    def image(self, rgb):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected one H×W×3 uint8 image')
        size = self.config['crop_size']
        stride = self.config['stride']
        h, w = rgb.shape[:2]
        # Mirror pad_full_image() without requiring or synthesizing a ground-truth mask.
        height = max(128, size, math.ceil(h / 64) * 64)
        width = max(128, size, math.ceil(w / 64) * 64)
        full = np.pad(rgb, ((0, height - h), (0, width - w), (0, 0)), mode='edge')
        thumb = Image.fromarray(full).resize((width // 4, height // 4), Image.Resampling.BOX)
        context = self.model.encode_global([self._normalize(np.asarray(thumb)).to(self.device)])[0]
        anchors, prior_components = _reference_regions(context[1].softmax(0).cpu().numpy(), h, w)
        image = self._normalize(full)
        del full
        coords = [(y, x) for y in self._positions(height, size, stride)
                  for x in self._positions(width, size, stride)]
        p = np.zeros((h, w, 4), dtype=np.float32)
        g = np.zeros_like(p)
        local = np.zeros((h, w, 8), dtype=np.float32)
        counts = np.zeros((h, w, 1), dtype=np.float32)
        captured = []

        def retain_head_input(module, args):
            captured.append(args[0][:, :8].permute(0, 2, 3, 1).contiguous().cpu().numpy())

        handle = self.model.head.register_forward_pre_hook(retain_head_input)
        try:
            batch = self.config['window_batch_size']
            for start in range(0, len(coords), batch):
                part = coords[start:start + batch]
                patches = torch.stack([image[:, y:y+size, x:x+size] for y, x in part]).to(self.device)
                boxes = torch.tensor([[y/height, x/width, (y+size)/height, (x+size)/width]
                                      for y, x in part], device=self.device)
                condition = self.model.encode_condition(patches, boxes, [context] * len(part))
                states = torch.zeros((len(part), self.state_channels, size, size), device=self.device)
                times = torch.zeros(len(part), dtype=torch.long, device=self.device)
                probs = self.model.denoise(condition, states, times).softmax(1).permute(0, 2, 3, 1).cpu().numpy()
                prior = condition['prior'].permute(0, 2, 3, 1).cpu().numpy()
                if len(captured) != 1:
                    raise RuntimeError('Expected exactly one segmentation head call per window batch')
                loc = captured.pop()
                for k, (y, x) in enumerate(part):
                    nh, nw = min(size, h - y), min(size, w - x)
                    if min(nh, nw) <= 0:
                        continue
                    sl = np.s_[y:y+nh, x:x+nw]
                    p[sl] += probs[k, :nh, :nw]
                    g[sl] += prior[k, :nh, :nw]
                    local[sl] += loc[k, :nh, :nw]
                    counts[sl] += 1
                del patches, condition, states, times, probs, prior, loc
        finally:
            handle.remove()
        if np.min(counts) < 1:
            raise RuntimeError('Uncovered native pixels in sliding-window prediction')
        for arr in (p, g, local):
            arr /= counts
            if not np.isfinite(arr).all():
                raise FloatingPointError('Nonfinite Direct features')
        return dict(p=p, g=g, local=local, rgb=rgb.astype(np.float32)/255.,
                    anchors=anchors, prior_components=prior_components)

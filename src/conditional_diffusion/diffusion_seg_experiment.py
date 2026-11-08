"""Independent RGB-conditioned mask diffusion experiments.

Only train/validation splits are exposed. Geometry is shared by the local crop
and full-image condition. Training uses FP32, uniform time sampling and no EMA.
The direct control keeps the ordinal auxiliary loss and receives zero state/time.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import MaterialDataset, MEAN, STD, read_manifest
from .diffusion_seg_model import ConditionalMaskDenoiser
from .inference import image_seed, positions
from .mask_diffusion import GaussianMaskDiffusion, MaskStateCodec
from .metrics_impl import boundary_counts, confusion, summarize, summarize_boundaries
from .runtime import restore_rng, rng_state, seed_all, source_hashes


def validate_config(config):
    if config['method'] not in ('diffusion', 'direct'):
        raise ValueError('method must be diffusion or direct')
    codec = MaskStateCodec(config['representation'], class_order=config['class_order'])
    for name in ('epochs', 'batch_size', 'crop_size', 'stride', 'timesteps',
                 'sampling_steps', 'window_batch_size', 'val_every', 'crops_per_image'):
        value = config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config['crop_size'] % 64 or config['stride'] > config['crop_size']:
        raise ValueError('crop_size must be divisible by 64 and stride <= crop_size')
    if config['timesteps'] < 2 or config['sampling_steps'] > config['timesteps']:
        raise ValueError('Invalid diffusion schedule')
    if not 0 <= config['thin_crop_fraction'] <= 1:
        raise ValueError('Invalid thin crop fraction')
    for name in ('workers', 'boundary_tolerance', 'seed', 'sampling_seed'):
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if not 0 <= config['warmup_epochs'] < config['epochs']:
        raise ValueError('warmup_epochs must be between zero and epochs - 1')
    for name in ('lr', 'min_lr', 'grad_clip', 'max_run_gib'):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ('weight_decay', 'prior_loss_weight', 'ce_weight', 'dice_weight',
                 'reconstruction_weight', 'min_free_gib', 'photometric_gain', 'photometric_gamma'):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if config['min_lr'] > config['lr']:
        raise ValueError('min_lr must not exceed lr')
    vignette = config.get('photometric_vignette', 0.0)
    if isinstance(vignette, bool) or not isinstance(vignette, (int, float)) or not math.isfinite(vignette) or not 0 <= vignette <= 0.5:
        raise ValueError('photometric_vignette must be finite and in [0, 0.5]')
    flip_mode = config.get('flip_mode', 'both')
    if flip_mode not in ('both', 'horizontal'):
        raise ValueError('flip_mode must be both or horizontal')
    sigma_rel = config.get('ema_sigma_rel', None)
    if sigma_rel is not None:
        if isinstance(sigma_rel, bool) or not isinstance(sigma_rel, (int, float)) or not math.isfinite(sigma_rel) or not 0 < sigma_rel < 1:
            raise ValueError('ema_sigma_rel must be null or finite in (0, 1)')
    if config['method'] == 'direct' and codec.representation != 'ordinal':
        raise ValueError('The registered direct control uses ordinal auxiliary supervision')
    return codec


def normalize(rgb):
    image = torch.from_numpy(np.array(rgb, copy=True, order='C')).permute(2, 0, 1).float() / 255
    return (image - MEAN) / STD


def pad_full_image(rgb, mask, minimum):
    height, width = rgb.shape[:2]
    # The global encoder requires thumbnails of at least 32 pixels per axis.
    padded_h = max(128, minimum, math.ceil(height / 64) * 64)
    padded_w = max(128, minimum, math.ceil(width / 64) * 64)
    padding = ((0, padded_h - height), (0, padded_w - width))
    return (np.pad(rgb, (*padding, (0, 0)), mode='edge'),
            np.pad(mask, padding, mode='edge'))


def global_inputs(rgb, mask):
    height, width = mask.shape
    thumbnail = Image.fromarray(np.ascontiguousarray(rgb)).resize(
        (width // 4, height // 4), Image.Resampling.BOX)
    blocks = mask.reshape(height // 16, 16, width // 16, 16)
    soft = np.stack([(blocks == label).mean((1, 3)) for label in range(4)])
    return normalize(np.asarray(thumbnail)), torch.from_numpy(soft.astype(np.float32))


class SegmentationDataset(Dataset):
    """Deterministic whole-image augmentation followed by aligned local crops.

    Validation keeps the native target size. Padding is restricted to the RGB
    condition and is cropped off predictions before any metric computation.
    """

    def __init__(self, config, split='train', allow_test=False):
        allowed = ('train', 'val', 'test') if allow_test else ('train', 'val')
        if split not in allowed:
            raise ValueError('This experiment exposes train and val only')
        self.config, self.split, self.epoch = config, split, 0
        self.source = MaterialDataset(config['manifest'], split, augment=False)

    def __len__(self):
        repeats = self.config['crops_per_image'] if self.split == 'train' else 1
        return len(self.source) * repeats

    def __getitem__(self, index):
        config = self.config
        training = self.split == 'train'
        source_index = index // config['crops_per_image'] if training else index
        rgb, mask, ident = self.source.raw(source_index)
        native_mask = mask.copy()
        generator = np.random.default_rng(np.random.SeedSequence([config['seed'], self.epoch, index]))
        if training:
            values = rgb.astype(np.float32) / 255
            gamma = np.exp(generator.uniform(-config['photometric_gamma'], config['photometric_gamma']))
            gain = 1 + generator.uniform(-config['photometric_gain'], config['photometric_gain'], 3)
            vignette = config.get('photometric_vignette', 0.0)
            if vignette > 0:
                values = values ** float(gamma) * gain.astype(np.float32)
                height_raw, width_raw = values.shape[:2]
                cy = generator.uniform(.3, .7) * height_raw
                cx = generator.uniform(.3, .7) * width_raw
                yy = ((np.arange(height_raw, dtype=np.float32) - cy) / (height_raw / 2))[:, None]
                xx = ((np.arange(width_raw, dtype=np.float32) - cx) / (width_raw / 2))[None, :]
                values = values * (1 - generator.uniform(0, vignette) * (yy ** 2 + xx ** 2) / 2)[:, :, None]
                rgb = (np.clip(values, 0, 1) * 255 + .5).astype(np.uint8)
            else:
                rgb = (np.clip(values ** gamma * gain, 0, 1) * 255 + .5).astype(np.uint8)
            turns = int(generator.integers(4))
            rgb, mask = np.rot90(rgb, turns), np.rot90(mask, turns)
            if config.get('flip_mode', 'both') == 'horizontal':
                if generator.random() < .5:
                    rgb, mask = np.ascontiguousarray(rgb[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])
            else:
                for axis in (0, 1):
                    if generator.random() < .5:
                        rgb, mask = np.flip(rgb, axis), np.flip(mask, axis)
        rgb, mask = pad_full_image(rgb, mask, config['crop_size'])
        thumbnail, soft = global_inputs(rgb, mask)
        height, width = mask.shape
        item = dict(thumb=thumbnail, soft=soft, id=ident)
        if not training:
            item.update(image=normalize(rgb), mask=torch.from_numpy(native_mask.astype(np.int64)))
            return item
        size = config['crop_size']
        thin = np.empty(0, dtype=np.int64)
        if generator.random() < config['thin_crop_fraction']:
            thin = np.flatnonzero(((mask == 1) | (mask == 2)).ravel()[::7]) * 7
        if len(thin):
            center_y, center_x = divmod(int(generator.choice(thin)), width)
            offset = generator.integers(-size // 4, size // 4 + 1, 2)
            top = int(np.clip(center_y - size // 2 + offset[0], 0, height - size))
            left = int(np.clip(center_x - size // 2 + offset[1], 0, width - size))
        else:
            top, left = int(generator.integers(height - size + 1)), int(generator.integers(width - size + 1))
        item.update(image=normalize(rgb[top:top + size, left:left + size]),
                    mask=torch.from_numpy(mask[top:top + size, left:left + size].astype(np.int64)),
                    box=torch.tensor([top / height, left / width, (top + size) / height,
                                      (left + size) / width], dtype=torch.float32))
        return item


def collate(items):
    result = {name: torch.stack([item[name] for item in items]) for name in ('image', 'mask', 'box')}
    result.update(thumb=[item['thumb'] for item in items], soft=[item['soft'] for item in items])
    return result


def configure(config):
    validate_config(config)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed_all(config['seed'])
    torch.set_num_threads(config.get('cpu_threads', 4))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(config['device']).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device is unavailable')


def build(config):
    codec = validate_config(config)
    model = ConditionalMaskDenoiser(state_channels=codec.channels, timesteps=config['timesteps'],
                                   **config.get('model', {})).to(config['device'])
    diffusion = GaussianMaskDiffusion(codec, config['timesteps'], config['device'])
    return model, diffusion


def prior_loss(contexts, targets):
    values = []
    for (_, logits, _), target in zip(contexts, targets):
        target = target.to(logits.device)
        if target.shape != logits.shape:
            raise ValueError('Global prior and aligned soft target shapes differ')
        ce = -(target * logits.log_softmax(0)).sum(0).mean()
        probabilities = logits.softmax(0)
        hard = F.one_hot(target.argmax(0), 4).permute(2, 0, 1).float()
        dice = 1 - ((2 * (probabilities * hard).sum((1, 2)) + 1e-6) /
                    (probabilities.sum((1, 2)) + hard.sum((1, 2)) + 1e-6))[1:].mean()
        values.append(.5 * ce + .5 * dice)
    return torch.stack(values).mean()


def train_step(model, diffusion, optimizer, batch, config):
    device = config['device']
    image, mask, boxes = [batch[name].to(device) for name in ('image', 'mask', 'box')]
    optimizer.zero_grad(set_to_none=True)
    contexts = model.encode_global([thumbnail.to(device) for thumbnail in batch['thumb']])
    condition = model.encode_condition(image, boxes, contexts)
    clean = diffusion.encode(mask)
    # Direct consumes the same draws as ordinal diffusion, then uses zero input.
    # This preserves subsequent RNG alignment for the matched training control.
    times = torch.randint(diffusion.timesteps, (len(image),), device=device)
    noise = torch.randn_like(clean)
    state = diffusion.q_sample(clean, times, noise)
    if config['method'] == 'direct':
        state, times = torch.zeros_like(clean), torch.zeros_like(times)
    logits = model.denoise(condition, state, times)
    terms = diffusion.loss(logits, mask, config['ce_weight'], config['dice_weight'],
                           config['reconstruction_weight'])
    terms['loss_prior'] = prior_loss(contexts, batch['soft'])
    loss = terms['loss_total'] + config['prior_loss_weight'] * terms['loss_prior']
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    terms['loss_total'] = loss
    return {name: float(value.detach()) for name, value in terms.items()}, float(gradient)


@torch.no_grad()
def predict_image(model, diffusion, image, thumbnail, ident, config, native_shape=None,
                  sampling_steps=None, sampling_seed=None):
    """Only image-derived conditions enter prediction. The caller owns labels."""
    model.eval()
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    height, width = image.shape[-2:]
    native_h, native_w = (height, width) if native_shape is None else native_shape
    size = config['crop_size']
    if min(height, width) < size or native_h > height or native_w > width:
        raise ValueError('Use aligned full-image padding before prediction')
    steps = config['sampling_steps'] if sampling_steps is None else sampling_steps
    seed = config['sampling_seed'] if sampling_seed is None else sampling_seed
    coordinates = [(y, x) for y in positions(height, size, config['stride'])
                   for x in positions(width, size, config['stride'])]
    context = model.encode_global([thumbnail.to(device)])[0]
    noise = None
    if config['method'] == 'diffusion':
        generator = torch.Generator().manual_seed(image_seed(ident, seed))
        noise = torch.randn((diffusion.codec.channels, height, width), generator=generator)
    total, counts = torch.zeros((4, height, width)), torch.zeros((height, width))
    batch_size = config['window_batch_size']
    for start in range(0, len(coordinates), batch_size):
        part = coordinates[start:start + batch_size]
        patches = torch.stack([image[:, y:y + size, x:x + size] for y, x in part]).to(device)
        boxes = torch.tensor([[y / height, x / width, (y + size) / height, (x + size) / width]
                              for y, x in part], device=device)
        condition = model.encode_condition(patches, boxes, [context] * len(part))
        if config['method'] == 'direct':
            states = torch.zeros((len(part), diffusion.codec.channels, size, size), device=device)
            times = torch.zeros(len(part), dtype=torch.long, device=device)
            probabilities = model.denoise(condition, states, times).softmax(1)
        else:
            states = torch.stack([noise[:, y:y + size, x:x + size] for y, x in part]).to(device)
            probabilities = diffusion.sample(model, condition, states, steps)
        if not bool(torch.isfinite(probabilities).all()):
            raise FloatingPointError('Nonfinite window probabilities')
        for probability, (y, x) in zip(probabilities.cpu(), part):
            total[:, y:y + size, x:x + size] += probability
            counts[y:y + size, x:x + size] += 1
    if not bool((counts > 0).all()):
        raise RuntimeError('Uncovered pixels in window fusion')
    probabilities = (total / counts)[:, :native_h, :native_w]
    prediction = probabilities.argmax(0).numpy().astype(np.uint8)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    effective_steps = 1 if config['method'] == 'direct' else steps
    return prediction, dict(seconds=time.perf_counter() - started, windows=len(coordinates),
                            nfe=len(coordinates) * effective_steps, sampling_steps=effective_steps)


@torch.no_grad()
def evaluate(model, diffusion, config, split='val', sampling_steps=None, sampling_seed=None,
             allow_test=False, extra_tolerances=()):
    dataset = SegmentationDataset(config, split, allow_test=allow_test)
    rows, matrices, boundaries = [], [], []
    extra_borders = {tol: [] for tol in extra_tolerances}
    for item in dataset:
        target = item['mask'].numpy()
        prediction, timing = predict_image(model, diffusion, item['image'], item['thumb'], item['id'],
                                           config, native_shape=target.shape,
                                           sampling_steps=sampling_steps, sampling_seed=sampling_seed)
        matrix = confusion(target, prediction)
        border = boundary_counts(target, prediction, config['boundary_tolerance'])
        matrices.append(matrix)
        boundaries.append(border)
        row = dict(id=item['id'], confusion=matrix.tolist(), boundaries=border.tolist(), **timing)
        for tol in extra_tolerances:
            eb = boundary_counts(target, prediction, tol)
            extra_borders[tol].append(eb)
            row[f'boundaries_tol{tol}'] = eb.tolist()
        rows.append(row)
    matrix, border = np.sum(matrices, axis=0), np.sum(boundaries, axis=0)
    metrics = {**summarize_material(matrix), **summarize_boundaries(border)}
    for tol in extra_tolerances:
        eb_sum = np.sum(extra_borders[tol], axis=0)
        bmetrics = summarize_boundaries(eb_sum)
        metrics[f'boundary_f1_foreground_tol{tol}'] = bmetrics['boundary_f1_foreground']
        for name in ('background', 'monolayer', 'fewlayer', 'multilayer'):
            metrics[f'boundary_f1_{name}_tol{tol}'] = bmetrics[f'boundary_f1_{name}']
    return dict(metrics=metrics, images=rows,
                nfe=sum(row['nfe'] for row in rows), seconds=sum(row['seconds'] for row in rows),
                error_rate_definition='Original semantic order 0/1/2/3; mono-few and absolute class-distance >=2 errors divided by all native pixels',
                timing_scope='predict only; excludes disk IO, dataset normalization and metric computation')


def summarize_material(matrix):
    metrics = summarize(matrix)
    thin = [metrics['iou_monolayer'], metrics['iou_fewlayer']]
    metrics['thin_mean_iou'] = float(np.nanmean(thin)) if np.isfinite(thin).any() else float('nan')
    total = int(matrix.sum())
    if total == 0:
        raise ValueError('Material metrics require at least one labeled pixel')
    metrics['monolayer_fewlayer_error_rate'] = float((matrix[1, 2] + matrix[2, 1]) / total)
    cross_level = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]) >= 2
    metrics['cross_level_error_rate'] = float(matrix[cross_level].sum() / total)
    return metrics


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def provenance(config, project, include_test=False):
    manifest = read_manifest(config['manifest'])
    split_names = ('train', 'val', 'test') if include_test else ('train', 'val')
    return dict(source=source_hashes(project), manifest=digest(config['manifest']),
                splits={split: digest(manifest['splits'][split]['snapshot_path']) for split in split_names})


class RunStorage:
    def __init__(self, run, config):
        self.run, self.config = Path(run), config

    def check(self, additional=0):
        used = sum(path.stat().st_size for path in self.run.rglob('*') if path.is_file() and not path.is_symlink())
        if used + additional > self.config['max_run_gib'] * 1024 ** 3:
            raise RuntimeError('Run storage budget exceeded')
        if shutil.disk_usage(self.run).free - additional < self.config['min_free_gib'] * 1024 ** 3:
            raise RuntimeError('Free-space reserve would be violated')


def atomic_save(path, value, storage, checkpoint=False):
    path = Path(path)
    estimate = 1 << 20
    if checkpoint:
        weights = sum(tensor.numel() * tensor.element_size() for tensor in value['model'].values())
        estimate += weights * ((4 if 'optimizer' in value else 1) + (1 if 'ema' in value else 0))
    storage.check(estimate)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            if checkpoint:
                torch.save(value, stream)
            else:
                stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def json_finite(value):
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@contextlib.contextmanager
def run_lock(run):
    """An OS advisory lock releases after process death; its one-byte file stays."""
    run = Path(run)
    run.parent.mkdir(parents=True, exist_ok=True)
    with (run.parent / ('.' + run.name + '.lock')).open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another process owns run {run}') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class PowerEMA:
    """EDM2 power-function EMA profile with fixed sigma_rel (online; no post-hoc snapshots)."""
    def __init__(self, model, sigma_rel):
        self.gamma = self.solve(sigma_rel)
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)
        self.n = 0

    @staticmethod
    def solve(sigma_rel):
        lo, hi = 0., 1000.
        for _ in range(200):
            g = (lo + hi) / 2
            s = math.sqrt((g + 1) / ((g + 2) ** 2 * (g + 3)))
            lo, hi = (g, hi) if s > sigma_rel else (lo, g)
        return (lo + hi) / 2

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        beta = (1 - 1 / self.n) ** (self.gamma + 1)
        for e, p in zip(self.shadow.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - beta)
        for e, p in zip(self.shadow.buffers(), model.buffers()):
            e.copy_(p)

    def state_dict(self):
        return dict(n=self.n, gamma=self.gamma, model=self.shadow.state_dict())

    def load_state_dict(self, state):
        self.n = state['n']
        self.shadow.load_state_dict(state['model'])


def scheduler_for(optimizer, config):
    def multiplier(epoch):
        warmup = config['warmup_epochs']
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = min(1., (epoch - warmup) / max(1, config['epochs'] - warmup - 1))
        floor = config['min_lr'] / config['lr']
        return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train(config, run, project, resume=False, stop_after_epoch=None, max_batches=None):
    configure(config)
    run, project = Path(run).resolve(), Path(project).resolve()
    if not run.is_relative_to(project / 'outputs') or run == project / 'outputs':
        raise ValueError('Runs must be named subdirectories of project/outputs')
    stop = config['epochs'] if stop_after_epoch is None else stop_after_epoch
    if not 1 <= stop <= config['epochs']:
        raise ValueError('stop_after_epoch must lie within the configured schedule')
    with run_lock(run):
        if resume:
            if (run / 'last.pt').exists():
                state = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
                if state['config'] != config or state['provenance'] != provenance(config, project):
                    raise ValueError('Resume configuration, source or split hashes changed')
            else:
                recover_initialization(run, config, project)
                state = None
        else:
            run.mkdir(parents=True, exist_ok=False)
            state = None
        storage = RunStorage(run, config)
        model, diffusion = build(config)
        ema_sigma_rel = config.get('ema_sigma_rel', None)
        ema = None if ema_sigma_rel is None else PowerEMA(model, ema_sigma_rel)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = scheduler_for(optimizer, config)
        frozen = provenance(config, project)
        epoch0, step, history, best = 0, 0, [], None
        if state is not None:
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            if ema is not None:
                if 'ema' not in state:
                    raise ValueError('EMA checkpoint state is missing')
                ema.load_state_dict(state['ema'])
            epoch0, step, history, best = state['epoch'], state['global_step'], state['history'], state['best']
            restore_rng(state['rng'])
        atomic_save(run / 'config.json', config, storage)
        atomic_save(run / 'provenance.json', frozen, storage)
        aug_desc = ('whole-image gamma/gain' +
                    (', vignette' if config.get('photometric_vignette', 0.0) > 0 else '') +
                    f", rotations and {config.get('flip_mode', 'both')} flips; edge pad to multiple of 64")
        atomic_save(run / 'metadata.json', dict(parameter_count=sum(p.numel() for p in model.parameters()),
                    precision='FP32; TF32 disabled',
                    ema=False if ema_sigma_rel is None else dict(type='PowerEMA', sigma_rel=ema_sigma_rel),
                    direct_control='zero state/time; same ordinal reconstruction loss',
                    augmentation=aug_desc,
                    validation='val only; best selected by four-class mIoU'), storage)

        def commit(epoch):
            if provenance(config, project) != frozen:
                raise ValueError('Source or split hashes changed during the run')
            payload = dict(config=config, provenance=frozen, model=model.state_dict(), optimizer=optimizer.state_dict(),
                           scheduler=scheduler.state_dict(), rng=rng_state(), epoch=epoch, global_step=step,
                           history=history, best=best)
            if ema is not None:
                payload['ema'] = ema.state_dict()
            atomic_save(run / 'last.pt', payload, storage, checkpoint=True)
            # last.pt is the authoritative commit, including the selected weights.
            # A resumed run repairs best.pt/history after any interrupted export.
            if best is not None:
                atomic_save(run / 'best.pt', dict(config=config, provenance=frozen, **best), storage, checkpoint=True)
            atomic_save(run / 'history.json', json_finite(history), storage)

        commit(epoch0)
        dataset = SegmentationDataset(config, 'train')
        for epoch in range(epoch0 + 1, stop + 1):
            if provenance(config, project) != frozen:
                raise ValueError('Source or split hashes changed during the run')
            dataset.epoch = epoch
            loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['workers'],
                                collate_fn=collate, generator=torch.Generator().manual_seed(config['seed'] + epoch))
            model.train()
            sums, seen = {}, 0
            started = time.perf_counter()
            learning_rate = optimizer.param_groups[0]['lr']
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                terms, gradient = train_step(model, diffusion, optimizer, batch, config)
                if ema is not None:
                    ema.update(model)
                count = len(batch['image'])
                seen += count
                step += 1
                for key, value in dict(**terms, gradient_norm=gradient).items():
                    sums[key] = sums.get(key, 0.) + value * count
            row = dict(epoch=epoch, global_step=step, lr=learning_rate, train_seconds=time.perf_counter() - started,
                       train={key: value / seen for key, value in sums.items()})
            if epoch % config['val_every'] == 0 or epoch == config['epochs']:
                eval_model = model if ema is None else ema.shadow
                row['validation'] = evaluate(eval_model, diffusion, config)
                score = row['validation']['metrics']['miou_4class']
                if best is None or score > best['validation']['metrics']['miou_4class']:
                    best = dict(epoch=epoch, global_step=step, validation=row['validation'],
                                model={key: value.detach().cpu().clone() for key, value in eval_model.state_dict().items()})
            scheduler.step()
            history.append(row)
            commit(epoch)
            print(json.dumps(dict(epoch=epoch, global_step=step, loss=row['train']['loss_total']), allow_nan=False), flush=True)
        return dict(epoch=max(epoch0, stop), global_step=step, best_epoch=None if best is None else best['epoch'])


def recover_initialization(run, config, project):
    """An explicit resume may retry an initialization that never committed epoch 0."""
    if not run.is_dir():
        raise FileNotFoundError(f'No run to resume: {run}')
    initial_names = ('config.json', 'provenance.json', 'metadata.json', 'last.pt')
    for path in run.iterdir():
        known = path.name in initial_names or any(
            path.name.startswith(name + '.') and path.name.endswith('.tmp') for name in initial_names)
        if not path.is_file() or not known:
            raise RuntimeError('Incomplete run has unexpected files; preserving it for inspection')
    for name, expected in (('config.json', config), ('provenance.json', provenance(config, project))):
        path = run / name
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != expected:
            raise ValueError('Incomplete initialization has different configuration or provenance')


def evaluate_checkpoint(config, run, project):
    configure(config)
    run = Path(run)
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    if state['config'] != config or state['provenance'] != provenance(config, project):
        raise ValueError('Evaluation configuration, source or splits differ from checkpoint')
    model, diffusion = build(config)
    model.load_state_dict(state['model'])
    return dict(epoch=state['epoch'], **evaluate(model, diffusion, config))


def smoke(config):
    """One real example/update and DDIM2, or the zero-state direct prediction."""
    config = dict(config, crop_size=64, stride=64, batch_size=1, workers=0,
                  sampling_steps=2, window_batch_size=1, crops_per_image=1)
    configure(config)
    dataset = SegmentationDataset(config, 'train')
    item = dataset[0]
    model, diffusion = build(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    model.train()
    terms, gradient = train_step(model, diffusion, optimizer, collate([item]), config)
    model.eval()
    with torch.no_grad():
        contexts = model.encode_global([item['thumb'].to(config['device'])])
        condition = model.encode_condition(item['image'][None].to(config['device']),
                                           item['box'][None].to(config['device']), contexts)
        noise = torch.randn((1, diffusion.codec.channels, 64, 64), device=config['device'])
        steps = 2
        if config['method'] == 'direct':
            times = torch.zeros(1, dtype=torch.long, device=config['device'])
            probabilities = model.denoise(condition, torch.zeros_like(noise), times).softmax(1)
            steps = 1
        else:
            probabilities = diffusion.sample(model, condition, noise, steps=steps)
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError('Nonfinite smoke sampling output')
    return dict(status='passed', source_id=item['id'], crop_size=64, optimizer_updates=1,
                method=config['method'], sampling_steps=steps, probabilities_shape=list(probabilities.shape),
                probability_sum_error=float((probabilities.sum(1) - 1).abs().max()),
                gradient_norm=gradient, losses=terms, checkpoint_saved=False)

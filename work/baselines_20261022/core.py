"""Training, inference, and evaluation core for the 9 Chapter-3 baseline methods.

Reuses GLPDataset, PowerEMA, metrics, positions, atomic_json, StorageGuard, and atomic_checkpoint
from glpflow_core and conditional_diffusion.runtime without modifying any historical files.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import math
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

WORK_DIR = Path(__file__).resolve().parents[1]
ROOT = WORK_DIR.parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import glpflow_core as G
import diagnose_glpflow_20261022 as D
from conditional_diffusion.inference import positions, preview
from conditional_diffusion.metrics_impl import confusion, summarize, boundary_counts, summarize_boundaries
from conditional_diffusion.runtime import (
    seed_all,
    rng_state,
    restore_rng,
    StorageGuard,
    atomic_checkpoint,
    ExperimentLogger,
    directory_size,
)

from .models import (
    UNet,
    UNetPlusPlus,
    PSPNet,
    DeepLabV3Plus,
    HRNetW48,
    SegFormerB1,
    Mask2Former,
)
from .diffusion import DDP, CCDM

BATCH = ROOT / 'outputs/launches/p1_baselines_20261022'
RUN_ROOT = ROOT / 'outputs/P1/baselines/seed_42'
TEST_OUT = ROOT / 'outputs/inference/baselines_test_20261022'

ORDER = [
    'unet',
    'unetpp',
    'deeplabv3p',
    'pspnet',
    'segformer_b1',
    'hrnet_w48',
    'mask2former',
    'ddp',
    'ccdm',
]

METHODS = {
    'unet': dict(display='U-Net', is_diffusion=False, nfe_per_window=1, grad_accum_steps=1),
    'unetpp': dict(display='UNet++', is_diffusion=False, nfe_per_window=1, grad_accum_steps=1),
    'deeplabv3p': dict(display='DeepLabv3+', is_diffusion=False, nfe_per_window=1, grad_accum_steps=1),
    'pspnet': dict(display='PSPNet', is_diffusion=False, nfe_per_window=1, grad_accum_steps=1),
    'segformer_b1': dict(display='SegFormer-B1', is_diffusion=False, nfe_per_window=1, grad_accum_steps=1),
    'hrnet_w48': dict(display='HRNet-W48', is_diffusion=False, nfe_per_window=1, grad_accum_steps=1),
    'mask2former': dict(display='Mask2Former', is_diffusion=False, nfe_per_window=1, grad_accum_steps=1),
    # DDP/CCDM: 2x4 accumulation, selected by check 3 of the 2026-10-22 engineering verification
    # (batch-8 micro-batches exceeded 22.5 GiB or ran out of memory; revision 2026-10-23).
    'ddp': dict(display='DDP', is_diffusion=True, sampling_steps=3, nfe_per_window=3, grad_accum_steps=2),
    'ccdm': dict(display='CCDM', is_diffusion=True, time_steps=250, sampling_steps=10, nfe_per_window=10, grad_accum_steps=2),
}

RADII = (2, 4, 8, 16)
STRATA = ['0-2', '2-4', '4-8', '8-16', '>16']


def distance_strata(truth: np.ndarray) -> np.ndarray:
    lab = np.full(truth.shape, len(RADII), np.uint8)
    for i, r in reversed(list(enumerate(RADII))):
        lab[D.band(truth, r)] = i
    return lab


def config(method: str) -> dict:
    assert method in METHODS, f"Unknown method: {method}"
    spec = METHODS[method]
    c = dict(
        method=method,
        arm=method,
        display=spec['display'],
        is_diffusion=spec['is_diffusion'],
        nfe_per_window=spec['nfe_per_window'],
        grad_accum_steps=spec['grad_accum_steps'],
        schema='baselines_v1',
        seed=42,
        sampling_seed=3407,
        manifest=str(ROOT / 'data_manifests/source_dataset.json'),
        device='cuda',
        crop_size=512,
        stride=384,
        crops_per_image=2,
        thin_crop_fraction=0.5,
        batch_size=8,
        workers=12,
        epochs=100,
        warmup_epochs=5,
        lr=2.8e-4,
        min_lr=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        val_every=10,
        window_batch_size=8,
        ema_sigma_rel=0.10,
        photometric=dict(gain=0.10, gamma=0.10, vignette=0.15),
        boundary_tolerance=2,
        preview_count=4,
        log_every=20,
        precision='FP32, TF32 disabled, deterministic algorithms enabled',
        data_scope='original train120/val25 (m109/m31 kept); test27 evaluated once at the end',
        storage=dict(
            min_free_gib=8,
            warning_free_gib=12,
            project_cap_gib=26,
            run_cap_gib=2.2,         # allows temporary .tmp coexistence during atomic_checkpoint
            steady_run_cap_gib=1.5,  # steady-state single-run disk cap per Section 5
            tensorboard_cap_mib=256,
        ),
    )
    if 'sampling_steps' in spec:
        c['sampling_steps'] = spec['sampling_steps']
    if 'time_steps' in spec:
        c['time_steps'] = spec['time_steps']
    return c


def build_model(c: dict) -> nn.Module:
    m = c['method']
    if m == 'unet':
        return UNet(num_classes=4)
    if m == 'unetpp':
        return UNetPlusPlus(num_classes=4)
    if m == 'pspnet':
        return PSPNet(num_classes=4)
    if m == 'deeplabv3p':
        return DeepLabV3Plus(num_classes=4)
    if m == 'hrnet_w48':
        return HRNetW48(num_classes=4)
    if m == 'segformer_b1':
        return SegFormerB1(num_classes=4)
    if m == 'mask2former':
        return Mask2Former(num_classes=4)
    if m == 'ddp':
        return DDP(num_classes=4, timesteps=c.get('sampling_steps', 3))
    if m == 'ccdm':
        return CCDM(num_classes=4, time_steps=c.get('time_steps', 250), sampling_steps=c.get('sampling_steps', 10))
    raise ValueError(f"Unknown method: {m}")


def build(c: dict, device: str = 'cuda'):
    G.configure(c['seed'])
    model = build_model(c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, G.lr_lambda(c))
    return model, opt, sched, G.PowerEMA(model, c['ema_sigma_rel'])


def make_loader(ds, c: dict, epoch: int):
    ds.epoch = epoch
    g = torch.Generator().manual_seed(c['seed'] * 1000 + epoch)
    return DataLoader(
        ds,
        batch_size=c['batch_size'],
        shuffle=True,
        num_workers=c['workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=G.collate,
        generator=g,
        persistent_workers=False,
    )


def train_step(model: nn.Module, opt: torch.optim.Optimizer, batch: dict, c: dict, device: str = 'cuda'):
    image_all = batch['image'].to(device)
    mask_all = batch['mask'].to(device)
    b_total = image_all.shape[0]
    accum = int(c.get('grad_accum_steps', 1))
    assert b_total % accum == 0, f"batch_size {b_total} not divisible by grad_accum_steps {accum}"
    mb = b_total // accum

    opt.zero_grad(set_to_none=True)
    agg_terms = {}
    for i in range(accum):
        img = image_all[i * mb : (i + 1) * mb]
        msk = mask_all[i * mb : (i + 1) * mb]
        _, terms = model.forward_train(img, msk)
        loss = terms['loss_total'] / accum
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss in {c['method']}")
        loss.backward()
        for k, v in terms.items():
            agg_terms[k] = agg_terms.get(k, 0.0) + float(v.detach()) / accum

    grad = nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'], error_if_nonfinite=True)
    opt.step()
    return agg_terms, float(grad)


@torch.no_grad()
def predict(model: nn.Module, c: dict, image: torch.Tensor, ident: str, steps: int = None,
            sampling_seed: int = None, return_prob: bool = False):
    """Only RGB image and image ID enter prediction; no ground truth argument."""
    device = next(model.parameters()).device
    steps = c.get('sampling_steps', 1) if steps is None else steps
    sampling_seed = c['sampling_seed'] if sampling_seed is None else sampling_seed
    h, w = image.shape[-2:]
    sz = min(c['crop_size'], h, w)
    stride = min(c['stride'], sz)
    m = c['method']

    ddp_field = None
    ccdm_field = None
    if m == 'ddp':
        seed_int = int.from_bytes(hashlib.sha256(f'{ident}:{sampling_seed}'.encode()).digest()[:8], 'little') % (2**63 - 1)
        gen = torch.Generator().manual_seed(seed_int)
        # Full-image noise grid at 1/4 resolution (with safety margin for window coords)
        ddp_field = torch.randn((256, h // 4 + 4, w // 4 + 4), generator=gen)
    elif m == 'ccdm':
        seed_int = int.from_bytes(hashlib.sha256(f'{ident}:{sampling_seed}'.encode()).digest()[:8], 'little') % (2**63 - 1)
        gen = torch.Generator().manual_seed(seed_int)
        ccdm_field = torch.rand((steps, h, w), generator=gen)

    coords = [(y, x) for y in positions(h, sz, stride) for x in positions(w, sz, stride)]
    total = torch.zeros((4, h, w))
    counts = torch.zeros((h, w))
    nfe = 0
    wb = c['window_batch_size']

    for start in range(0, len(coords), wb):
        part = coords[start : start + wb]
        b = len(part)
        img = torch.stack([image[:, y : y + sz, x : x + sz] for y, x in part]).to(device)
        if m == 'ddp':
            hs, ws = sz // 4, sz // 4
            eps = torch.stack([ddp_field[:, y // 4 : y // 4 + hs, x // 4 : x // 4 + ws] for y, x in part]).to(device)
            logits, calls = model.infer_windows(img, eps=eps, steps=steps)
        elif m == 'ccdm':
            u_win = torch.stack([ccdm_field[:, y : y + sz, x : x + sz] for y, x in part]).to(device)
            logits, calls = model.infer_windows(img, u_field=u_win, steps=steps)
        else:
            logits = model(img)
            calls = b
        nfe += calls
        prob = logits.softmax(1).cpu()
        for pr, (y, x0) in zip(prob, part):
            total[:, y : y + sz, x0 : x0 + sz] += pr
            counts[y : y + sz, x0 : x0 + sz] += 1

    assert bool((counts > 0).all())
    prob = total / counts
    pred = prob.argmax(0).numpy().astype(np.uint8)
    return (pred, nfe, prob) if return_prob else (pred, nfe)


@torch.no_grad()
def evaluate(model: nn.Module, c: dict, split: str = 'val', steps: int = None,
             sampling_seed: int = None, limit: int = None, on_image=None):
    model.eval()
    ds = G.GLPDataset(c, split, train=False)
    n = len(ds) if limit is None else min(limit, len(ds))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_wall = time.perf_counter()

    matrices, bounds, ids, previews = [], [], [], {}
    nfe = 0
    pred_secs = []
    ok_strata = np.zeros((len(STRATA), 4), np.int64)
    n_strata = np.zeros((len(STRATA), 4), np.int64)

    for i in range(n):
        d = ds[i]
        torch.cuda.synchronize()
        t_pred0 = time.perf_counter()
        pred, calls = predict(model, c, d['image'], d['id'], steps=steps, sampling_seed=sampling_seed)
        torch.cuda.synchronize()
        pred_secs.append(time.perf_counter() - t_pred0)

        truth = d['mask'].numpy()
        nfe += calls
        matrices.append(confusion(truth, pred))
        bounds.append(boundary_counts(truth, pred, c['boundary_tolerance']))
        ids.append(d['id'])

        lab = distance_strata(truth)
        hit = (pred == truth)
        for s_idx in range(len(STRATA)):
            for k in range(4):
                sel = (lab == s_idx) & (truth == k)
                n_strata[s_idx, k] += int(sel.sum())
                ok_strata[s_idx, k] += int(hit[sel].sum())

        if i < c['preview_count']:
            previews[d['id']] = preview(d['image'], truth, pred)
        if on_image is not None:
            on_image(d['id'], pred)

    torch.cuda.synchronize()
    m = np.sum(matrices, axis=0)
    bc = np.sum(bounds, axis=0)
    dist_acc = {
        s_name: dict(
            all=float(ok_strata[idx].sum() / max(1, n_strata[idx].sum())),
            thin=float(ok_strata[idx, 1:3].sum() / max(1, n_strata[idx, 1:3].sum())),
            pixels=int(n_strata[idx].sum()),
            thin_pixels=int(n_strata[idx, 1:3].sum()),
        )
        for idx, s_name in enumerate(STRATA)
    }
    metrics = {
        **G.classification_metrics(m),
        **summarize_boundaries(bc),
        'distance_accuracy': dist_acc,
        'acc_0_2px_all': dist_acc['0-2']['all'],
        'acc_0_2px_thin': dist_acc['0-2']['thin'],
        'seconds': time.perf_counter() - start_wall,
        'predict_seconds_per_image': float(np.mean(pred_secs)),
        'predict_seconds_per_image_excl_first': float(np.mean(pred_secs[1:])) if len(pred_secs) > 1 else float(np.mean(pred_secs)),
        'image_count': len(ids),
        'nfe_total': int(nfe),
        'nfe_per_image': float(nfe / max(1, len(ids))),
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
    }
    mats = dict(
        ids=np.array(ids),
        per_image=np.array(matrices),
        global_matrix=m,
        boundary_counts=bc,
        per_image_boundary=np.array(bounds),
        strata_correct=ok_strata,
        strata_total=n_strata,
    )
    return metrics, mats, previews


def checkpoint_payload(model, opt, sched, ema, c, epoch, step, best, plan_sha):
    return dict(
        model=model.state_dict(),
        optimizer=opt.state_dict(),
        scheduler=sched.state_dict(),
        ema=ema.state_dict(),
        rng=rng_state(),
        config=c,
        epoch=epoch,
        global_step=step,
        best_miou=best,
        plan_sha256=plan_sha,
    )


def _check_steady_cap(run: Path, c: dict):
    cap_gib = c['storage'].get('steady_run_cap_gib', 1.5)
    sz = directory_size(run)
    if sz > cap_gib * (1024**3):
        raise RuntimeError(f"Steady-state run directory size {sz / 1024**3:.3f} GiB exceeded {cap_gib} GiB cap")


def train(c: dict, run, plan_sha: str = 'unfrozen', max_batches: int = None,
          stop_after_epoch: int = None, guard_fn=None):
    run = Path(run)
    device = c['device']
    model, opt, sched, ema = build(c, device)
    last = run / 'checkpoints/last.pt'
    epoch0 = step = 0
    best = -1.0
    sg = StorageGuard(ROOT, run, c['storage'])

    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        assert state['config'] == c and state['plan_sha256'] == plan_sha, 'resume requires identical config/plan'
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        ema.load_state_dict(state['ema'])
        restore_rng(state['rng'])
        epoch0, step, best = state['epoch'], state['global_step'], state['best_miou']
        del state
    else:
        (run / 'checkpoints').mkdir(parents=True, exist_ok=True)
        G.atomic_json(run / 'config.json', c)
        n_params = sum(p.numel() for p in model.parameters())
        G.atomic_json(
            run / 'run_metadata.json',
            dict(
                parameters=n_params,
                fp32_bytes=4 * n_params,
                method=c['method'],
                arm=c['arm'],
                grad_accum_steps=c['grad_accum_steps'],
                effective_batch_size=c['batch_size'],
                bn_micro_batch_size=c['batch_size'] // c['grad_accum_steps'],
                plan_sha256=plan_sha,
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),
            ),
        )
        atomic_checkpoint(last, checkpoint_payload(model, opt, sched, ema, c, 0, 0, best, plan_sha), sg)
        _check_steady_cap(run, c)

    if epoch0 >= c['epochs']:
        return dict(status='completed', epoch=epoch0, best=best)

    ds = G.GLPDataset(c, 'train')
    logger = ExperimentLogger(run, sg, step if epoch0 else None, epoch0 if epoch0 else None)
    G.atomic_json(run / 'status.json', dict(status='running', last_completed_epoch=epoch0, global_step=step))

    try:
        for epoch in range(epoch0 + 1, c['epochs'] + 1):
            if guard_fn:
                guard_fn()
            model.train()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            sums, seen, lr = {}, 0, opt.param_groups[0]['lr']

            for bi, batch in enumerate(make_loader(ds, c, epoch)):
                if max_batches is not None and bi >= max_batches:
                    break
                terms, grad = train_step(model, opt, batch, c, device)
                ema.update(model)
                n = batch['image'].shape[0]
                seen += n
                step += 1
                for k, v in terms.items():
                    sums[k] = sums.get(k, 0.0) + v * n
                if step % c['log_every'] == 0:
                    scalar_Step = {'train_step/' + k: v for k, v in dict(**terms, grad_norm=grad, lr=lr).items()}
                    logger.scalars('steps', scalar_Step, epoch, step)
                    G.atomic_json(
                        run / 'progress.json',
                        dict(
                            epoch=epoch,
                            global_step=step,
                            loss=terms['loss_total'],
                            updated_at_utc=datetime.now(timezone.utc).isoformat(),
                        ),
                    )

            torch.cuda.synchronize()
            secs = time.perf_counter() - t0
            train_peak_mib = torch.cuda.max_memory_reserved() / 2**20
            vals = {'train_epoch/' + k: v / seen for k, v in sums.items()}
            vals.update({
                'resources/train_seconds': secs,
                'resources/samples_per_second': seen / secs,
                'resources/peak_reserved_mib': train_peak_mib,
                'optim/lr': lr,
            })
            logger.scalars('epochs', vals, epoch, step)
            metrics = None

            if max_batches is None and (epoch % c['val_every'] == 0 or epoch == c['epochs']):
                metrics, mats, previews = evaluate(ema.shadow, c)
                metrics['train_peak_reserved_mib'] = train_peak_mib
                scalar_val = {
                    'val/' + k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float, np.number))
                }
                logger.scalars('val', scalar_val, epoch, step)
                logger.images(previews, epoch)
                G.atomic_json(run / f'validation_epoch{epoch}.json', metrics)
                if metrics['miou_4class'] > best:
                    best = metrics['miou_4class']
                    atomic_checkpoint(
                        run / 'checkpoints/best.pt',
                        dict(
                            model=ema.shadow.state_dict(),
                            config=c,
                            epoch=epoch,
                            global_step=step,
                            validation_metrics=metrics,
                            plan_sha256=plan_sha,
                            weights='power_ema',
                        ),
                        sg,
                    )
                    G.atomic_json(run / 'best_val_metrics.json', dict(epoch=epoch, metrics=metrics))
                    np.savez_compressed(run / 'best_val_confusion.npz', **mats)

            sched.step()
            logger.flush()
            atomic_checkpoint(last, checkpoint_payload(model, opt, sched, ema, c, epoch, step, best, plan_sha), sg)
            _check_steady_cap(run, c)
            G.atomic_json(
                run / 'status.json',
                dict(
                    status='completed' if epoch == c['epochs'] else 'running',
                    last_completed_epoch=epoch,
                    global_step=step,
                    best_miou=best,
                ),
            )
            print(
                dict(
                    method=c['method'],
                    epoch=epoch,
                    step=step,
                    loss=round(sums['loss_total'] / seen, 5),
                    sec=round(secs, 1),
                    val=None if metrics is None else round(metrics['miou_4class'], 5),
                ),
                flush=True,
            )
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                break
    except BaseException as e:
        G.atomic_json(run / 'status.json', dict(status='interrupted', error=str(e), global_step=step))
        raise
    finally:
        logger.close()
    return dict(status='completed' if epoch == c['epochs'] else 'stopped', epoch=epoch, best=best)

"""Isolated S0/S1/S2 workflow. Historical trainers and checkpoints stay unchanged."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import MaterialDataset, read_manifest
from .diffusion import GaussianDiffusion
from .inference import positions, image_seed, preview
from .metrics_impl import confusion, summarize, boundary_counts, summarize_boundaries
from .objectives import losses
from .recycling import mixed_recycling_input
from .runtime import (seed_all, rng_state, restore_rng, source_hashes, write_json,
                      StorageGuard, ExperimentLogger, atomic_checkpoint, directory_size)
from .static_semantic import StaticSemanticModel, sample_cached


def configure(c):
    torch.set_num_threads(c['cpu_threads'])
    seed_all(c['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    # Match the registered historical FP32 trainer's cuDNN setting explicitly.
    torch.backends.cudnn.allow_tf32 = True


def build(c):
    return StaticSemanticModel(c['mode']).to(c['device'])


def train_step(model, diffusion, optimizer, image, target, c):
    optimizer.zero_grad(set_to_none=True)
    x0 = None
    counts = {}
    if c['mode'] == 'S2':
        x0 = diffusion.encode(target)
        t = diffusion.training_timesteps(len(image), 'high_noise_stratified', 900)
        noise = torch.randn_like(x0)
        noisy, selected = mixed_recycling_input(model, diffusion, image, x0, t, noise, .5)
        output = model(image, noisy, t)
        counts = dict(timestep_sum=int(t.sum()), high_noise_count=int((t >= 900).sum()),
                      recycled_count=selected.numel(), recycled_high_noise_count=int((t[selected] >= 900).sum()))
    else:
        output = model(image)
    loss, terms = losses(output, target, c, x0)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    return {k: float(v.detach()) for k, v in terms.items()}, float(grad), counts


def loader_for(dataset, c, epoch):
    dataset.epoch = epoch
    return DataLoader(dataset, batch_size=c['batch_size'], shuffle=True,
                      generator=torch.Generator().manual_seed(c['seed']+epoch),
                      num_workers=c['workers'], pin_memory=c['device'] == 'cuda', drop_last=False)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict(model, diffusion, image, ident, c, return_probabilities=False):
    """Whole-image noise, ordered probability fusion; no labels enter prediction."""
    device = next(model.parameters()).device
    h, w = image.shape[-2:]
    window, stride, batch_size = c['crop_size'], c['stride'], c['window_batch_size']
    ph, pw = max(h, window), max(w, window)
    clock = time.perf_counter()
    padded = F.pad(image[None], (0, pw-w, 0, ph-h), mode='replicate')[0]
    coords = [(y, x) for y in positions(ph, window, stride) for x in positions(pw, window, stride)]
    field = None
    if c['mode'] == 'S2':
        generator = torch.Generator().manual_seed(image_seed(ident, c['sampling_seed']))
        field = torch.randn((4, ph, pw), generator=generator)
    total, counts = torch.zeros((4, ph, pw)), torch.zeros((ph, pw))
    timing = dict(preparation_seconds=time.perf_counter()-clock, transfer_seconds=0.,
                  sampling_seconds=0., fusion_seconds=0.)
    for start in range(0, len(coords), batch_size):
        part = coords[start:start+batch_size]
        clock = time.perf_counter()
        patches = torch.stack([padded[:, y:y+window, x:x+window] for y, x in part]).to(device)
        noise = None if field is None else torch.stack([field[:, y:y+window, x:x+window] for y, x in part]).to(device)
        synchronize(device)
        timing['transfer_seconds'] += time.perf_counter()-clock
        clock = time.perf_counter()
        output = sample_cached(model, diffusion, patches, noise, c['sampling_steps']) if noise is not None else model(patches)
        probabilities = output.softmax(1)
        synchronize(device)
        timing['sampling_seconds'] += time.perf_counter()-clock
        clock = time.perf_counter()
        probabilities = probabilities.cpu()
        synchronize(device)
        timing['transfer_seconds'] += time.perf_counter()-clock
        clock = time.perf_counter()
        for probability, (y, x) in zip(probabilities, part):
            total[:, y:y+window, x:x+window] += probability
            counts[y:y+window, x:x+window] += 1
        timing['fusion_seconds'] += time.perf_counter()-clock
    if not bool((counts > 0).all()):
        raise RuntimeError('Uncovered image pixels')
    clock = time.perf_counter()
    probabilities = total[:, :h, :w]/counts[:h, :w]
    prediction = probabilities.argmax(0).numpy().astype(np.uint8)
    timing['fusion_seconds'] += time.perf_counter()-clock
    nfe = len(coords)*(1 if c['mode'] == 'S0' else 5)
    return prediction, nfe, timing, probabilities if return_probabilities else None


@torch.no_grad()
def evaluate_static(model, diffusion, dataset, c):
    model.eval()
    synchronize(c['device'])
    if c['device'] == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    matrices, ids, previews, breakdown = [], [], {}, []
    boundaries = np.zeros((4, 4), dtype=np.int64)
    calls = 0
    for i in range(len(dataset)):
        clock = time.perf_counter()
        image, target, ident = dataset[i]
        read_seconds = time.perf_counter()-clock
        pred, nfe, timing, _ = predict(model, diffusion, image, ident, c)
        clock = time.perf_counter()
        target = target.numpy()
        matrices.append(confusion(target, pred))
        if c['boundary_metrics']:
            boundaries += boundary_counts(target, pred, c['boundary_tolerance'])
        timing.update(id=ident, read_seconds=read_seconds, metrics_seconds=time.perf_counter()-clock)
        breakdown.append(timing)
        ids.append(ident)
        calls += nfe
        if i < c['preview_count']:
            previews[ident] = preview(image, target, pred)
    synchronize(c['device'])
    matrix = np.sum(matrices, axis=0)
    metrics = summarize(matrix)
    if c['boundary_metrics']:
        metrics.update(summarize_boundaries(boundaries))
    metrics.update(seconds=time.perf_counter()-begin, image_count=len(ids), nfe_total=calls)
    if c['device'] == 'cuda':
        metrics.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    return metrics, dict(ids=np.array(ids), per_image=np.array(matrices), global_matrix=matrix,
                         boundary_counts=boundaries), previews, breakdown


def scheduler_for(optimizer, c):
    def multiplier(epoch):
        if epoch < c['warmup_epochs']:
            return (epoch+1)/max(1, c['warmup_epochs'])
        progress = min(1., (epoch-c['warmup_epochs'])/max(1, c['epochs']-c['warmup_epochs']-1))
        minimum = c['min_lr']/c['lr']
        return minimum+(1-minimum)*.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def restore_best_sidecars(run, payload):
    """The atomic best checkpoint is authoritative after an interrupted commit."""
    matrices = payload['validation_matrices']
    temp = Path(run)/'best_val_confusion.recovery.npz'
    np.savez_compressed(temp, **matrices)
    temp.replace(Path(run)/'best_val_confusion.npz')
    write_json(Path(run)/'best_val_metrics.json', dict(epoch=payload['epoch'], metrics=payload['validation_metrics']))


def train_static(c, project, run, frozen_source, resume=False):
    project, run = Path(project), Path(run)
    configure(c)
    assert c['seed'] == 42 and c['epochs'] == 40 and c['crop_size'] == 512
    assert c['batch_size'] in [2, 4, 8, 16] and c['window_batch_size'] in [1, 4, 8, 16]
    assert c['sampling_steps'] == 5 and c['sampling_seed'] == 3407 and not c['boundary']
    assert source_hashes(project) == frozen_source, 'Source/config freeze mismatch'
    if resume and not (run/'checkpoints/last.pt').exists():
        # No optimization starts before the epoch0 checkpoint. Preserve even an
        # interrupted directory initialization instead of overwriting its files.
        assert not (run/'checkpoints/best.pt').exists(), 'Missing last with existing best requires explicit recovery'
        assert not (run/'metrics_steps.csv').exists(), 'Uncheckpointed updates require explicit recovery'
        if (run/'config.json').exists():
            assert json.loads((run/'config.json').read_text()) == c
        archived = run.with_name(run.name+'_incomplete_start_'+str(time.time_ns()))
        run.rename(archived)
        resume = False
    manifest = read_manifest(c['manifest'])
    training = MaterialDataset(c['manifest'], 'train', crop=c['crop_size'], seed=42)
    validation = MaterialDataset(c['manifest'], 'val', augment=False)
    assert len(training) == 120 and len(validation) == 25
    guard = StorageGuard(project, run, c['storage'])
    guard.check()
    model = build(c)
    parameters = sum(p.numel() for p in model.parameters())
    guard.check(max(0, 16*parameters+(.25+.25+.03125)*1024**3-directory_size(run))+12*parameters+8*1024**2)
    diffusion = GaussianDiffusion(1000, c['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    scheduler = scheduler_for(optimizer, c)
    epoch0, step, best = 0, 0, -float('inf')
    if resume:
        state = torch.load(run/'checkpoints/last.pt', map_location='cpu', weights_only=False)
        assert state['config'] == c and state['source_sha256'] == frozen_source
        for name in ['last.pt.tmp', 'best.pt.tmp']:
            pending = run/'checkpoints'/name
            if pending.exists():
                pending.rename(pending.with_name(pending.name+'.uncommitted_'+str(time.time_ns())))
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); restore_rng(state['rng'])
        epoch0, step, best = state['epoch'], state['global_step'], state['best_miou']
        if (run/'checkpoints/best.pt').exists():
            canonical_best = torch.load(run/'checkpoints/best.pt', map_location='cpu', weights_only=False)
            assert canonical_best['config'] == c and canonical_best['source_sha256'] == frozen_source
            restore_best_sidecars(run, canonical_best)
            best = max(best, canonical_best['validation_metrics']['miou_4class'])
    else:
        run.mkdir(parents=True, exist_ok=False)
        (run/'checkpoints').mkdir()
        write_json(run/'config.json', c)
        write_json(run/'run_metadata.json', dict(parameter_count=parameters, trainable_parameter_count=parameters,
                   source_sha256=frozen_source, split_hashes={k:v['sha256'] for k,v in manifest['splits'].items()},
                   torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                   train_records=120, train_unique_rgb=119, effective_exposures=4800,
                   planned_updates=40*math.ceil(120/c['batch_size']), initialization='seed42 separate fixed RGB+decoder and state RNG domains; no pretrained weights',
                   precision='FP32 tensors; deterministic; matmul TF32 false / cuDNN TF32 true (historical setting)',
                   training_budget='scratch40 seed42; benchmark weights discarded', mode=c['mode']))
        atomic_checkpoint(run/'checkpoints/last.pt', dict(model=model.state_dict(), config=c, optimizer=optimizer.state_dict(),
                          scheduler=scheduler.state_dict(), rng=rng_state(), epoch=0, global_step=0,
                          best_miou=best, source_sha256=frozen_source), guard)
    if epoch0 == 40:
        assert step == 40*math.ceil(120/c['batch_size'])
        write_json(run/'status.json', dict(status='completed', last_completed_epoch=40, global_step=step))
        return
    logger = ExperimentLogger(run, guard, step if resume else None, epoch0 if resume else None)
    write_json(run/'status.json', dict(status='running', last_completed_epoch=epoch0, global_step=step))
    try:
        for epoch in range(epoch0+1, 41):
            assert source_hashes(project) == frozen_source, 'Source/config changed during training'
            guard.check()
            loader = loader_for(training, c, epoch)
            model.train()
            synchronize(c['device']); torch.cuda.reset_peak_memory_stats()
            begin, seen, sums, counts = time.perf_counter(), 0, {}, {}
            lr = optimizer.param_groups[0]['lr']
            for image, target, _ in loader:
                image, target = image.to(c['device']), target.to(c['device'])
                terms, grad, tally = train_step(model, diffusion, optimizer, image, target, c)
                n = len(image); seen += n; step += 1
                for key, value in terms.items():
                    sums[key] = sums.get(key, 0.)+value*n
                for key, value in tally.items():
                    counts[key] = counts.get(key, 0)+value
                if step % c['log_every'] == 0:
                    logger.scalars('steps', {'train_step/'+k:v for k,v in dict(**terms, grad_norm_pre_clip=grad, lr=lr).items()}, epoch, step)
            synchronize(c['device'])
            seconds = time.perf_counter()-begin
            assert seen == 120 and step == epoch*math.ceil(120/c['batch_size'])
            scalars = {'train_epoch/'+k:v/seen for k,v in sums.items()}
            scalars.update({'resources/train_seconds':seconds, 'resources/train_samples_per_second':seen/seconds,
                            'resources/train_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                            'resources/train_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
                            'optim/epoch_lr_group_0':lr, 'budget/epoch_exposures':seen,
                            'budget/optimizer_updates':math.ceil(120/c['batch_size'])})
            if c['mode'] == 'S2':
                assert counts['high_noise_count'] == counts['recycled_count'] == 60
                scalars.update({'diffusion_train/'+k:v/seen for k,v in counts.items()})
            logger.scalars('epochs', scalars, epoch, step)
            val_metrics = None
            if epoch % c['val_every'] == 0:
                val_metrics, matrices, previews, breakdown = evaluate_static(model, diffusion, validation, c)
                assert val_metrics['image_count'] == 25 and val_metrics['nfe_total'] == (875 if c['mode'] == 'S0' else 4375)
                logger.scalars('val', {'val/'+k:v for k,v in val_metrics.items()}, epoch, step)
                logger.images(previews, epoch)
                np.savez_compressed(run/'latest_val_confusion.npz', **matrices)
                write_json(run/f'validation_timing_epoch{epoch}.json', breakdown)
                if val_metrics['miou_4class'] > best:
                    best = val_metrics['miou_4class']
                    payload = dict(model=model.state_dict(), config=c, source_sha256=frozen_source, epoch=epoch,
                                   global_step=step, validation_metrics=val_metrics, validation_matrices=matrices)
                    atomic_checkpoint(run/'checkpoints/best.pt', payload, guard)
                    restore_best_sidecars(run, payload)
            scheduler.step(); logger.flush()
            atomic_checkpoint(run/'checkpoints/last.pt', dict(model=model.state_dict(), config=c, optimizer=optimizer.state_dict(),
                              scheduler=scheduler.state_dict(), rng=rng_state(), epoch=epoch, global_step=step,
                              best_miou=best, source_sha256=frozen_source), guard)
            write_json(run/'status.json', dict(status='completed' if epoch == 40 else 'running', last_completed_epoch=epoch, global_step=step))
            print(dict(mode=c['mode'], epoch=epoch, global_step=step, train_loss=sums['loss_total']/seen,
                       train_seconds=seconds, val_miou=None if val_metrics is None else val_metrics['miou_4class']), flush=True)
    except BaseException as exc:
        write_json(run/'status.json', dict(status='interrupted', error=str(exc), global_step=step,
                   recovery='Resume only from last.pt with frozen identical source/config'))
        raise
    finally:
        logger.close()

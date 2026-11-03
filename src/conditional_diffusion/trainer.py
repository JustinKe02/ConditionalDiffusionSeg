import math
import hashlib
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MaterialDataset, FixedCropDataset, read_manifest, CLASSES, PALETTE
from .reference_data import dataset_from_config
from .model import make_model
from .diffusion import GaussianDiffusion
from .objectives import losses
from .recycling import mixed_recycling_input
from .inference import evaluate
from .runtime import (ExperimentLogger, StorageGuard, atomic_checkpoint, write_json,
                      seed_all, rng_state, restore_rng, source_hashes, directory_size)


def validate_config(c):
    if c.get('schema_version') != 1:
        raise ValueError('Expected runnable config schema_version=1')
    if c['method'] not in ['direct', 'iterative', 'diffusion']:
        raise ValueError('Unknown method')
    for k in ['epochs', 'batch_size', 'crop_size', 'stride', 'iterations', 'timesteps',
              'sampling_steps', 'samples_per_image', 'val_every', 'log_every', 'cpu_threads']:
        if not isinstance(c[k], int) or c[k] <= 0:
            raise ValueError(f'Invalid {k}')
    if c['crop_size'] < 16 or c['crop_size'] % 8 or c['base_channels'] < 8 or c['base_channels'] % 2:
        raise ValueError('Crop must be >=16 and divisible by 8; base must be even and >=8')
    if c['stride'] > c['crop_size'] or c['sampling_steps'] > c['timesteps']:
        raise ValueError('Invalid stride or sampler steps')
    if not c['debug'] and (c['train_limit'] is not None or c['val_limit'] is not None or c['max_batches'] is not None):
        raise ValueError('Subset runs require debug=true')
    if c['seed'] < 0 or c['sampling_seed'] < 0 or c['workers'] < 0:
        raise ValueError('Seeds/workers must be nonnegative')
    reference_variants = {'reference_fusion_f0': (False, 'mean'), 'reference_fusion_f1': (True, 'mean'),
                          'reference_fusion_f2': (False, 'adaptive'), 'reference_fusion_f3': (True, 'adaptive')}
    adapter_variants = {'reference_adapter_a': False, 'reference_adapter_b': True}
    adapter_mode = c.get('condition_architecture') == 'frozen_reference_adapter'
    if adapter_mode or c.get('experiment_variant') in adapter_variants:
        if not (adapter_mode and c.get('experiment_variant') in adapter_variants
                and isinstance(c.get('adapter_reference_enabled'), bool)
                and c['adapter_reference_enabled'] == adapter_variants[c['experiment_variant']]
                and c.get('condition_fusion') == 'output_residual'
                and c.get('reference_enabled') is True and c.get('reference_contrast_weighted') is False
                and c['method'] == 'diffusion' and c['phase'] == 'P1' and not c['debug']
                and not c.get('fixed_crops') and not c.get('dual_head', False)
                and c['base_channels'] == 32 and c.get('adapter_width') == 32
                and bool(c.get('adapter_base_checkpoint')) and isinstance(c.get('adapter_base_sha256'), str)
                and len(c['adapter_base_sha256']) == 64
                and bool(c.get('reference_cache')) and isinstance(c.get('reference_cache_sha256'), str)
                and len(c['reference_cache_sha256']) == 64):
            raise ValueError('Frozen reference adapter requires registered matched P1 settings')
    reference_keys = ['condition_architecture', 'condition_fusion', 'reference_enabled', 'reference_cache', 'reference_cache_sha256']
    if not adapter_mode and (c.get('experiment_variant') in reference_variants or any(k in c for k in reference_keys)):
        expected = reference_variants.get(c.get('experiment_variant'))
        if not (expected is not None and c.get('condition_architecture') == 'split_rgb_state'
                and isinstance(c.get('reference_enabled'), bool)
                and (c['reference_enabled'], c.get('condition_fusion')) == expected
                and c['method'] == 'diffusion' and c['phase'] == 'P1' and not c['debug']
                and not c.get('fixed_crops') and not c.get('dual_head', False)
                and bool(c.get('reference_cache')) and isinstance(c.get('reference_cache_sha256'), str)
                and len(c['reference_cache_sha256']) == 64):
            raise ValueError('Reference conditioning requires a registered P1 variant and matching frozen cache/features/fusion')
    policy = c.get('timestep_sampling', 'uniform')
    if policy not in ['uniform', 'high_noise_stratified']:
        raise ValueError('Unknown timestep sampling policy')
    if policy == 'high_noise_stratified':
        p0_ablation = c['phase'] == 'P0' and c['debug'] and bool(c.get('fixed_crops'))
        p1_ablation = (c['phase'] == 'P1' and not c['debug'] and not c.get('fixed_crops')
                       and c.get('experiment_variant') in ['high_noise_coverage', 'high_noise_recycling', 'high_noise_recycling_dual_head', *reference_variants, *adapter_variants])
        if c['method'] != 'diffusion' or not (p0_ablation or p1_ablation):
            raise ValueError('High-noise sampling requires fixed-crop P0 or explicitly registered P1 ablation')
        if c['batch_size'] < 2 or c['batch_size'] % 2 or not 0 < c.get('high_noise_start', 900) < c['timesteps']:
            raise ValueError('Invalid high-noise stratum configuration')
    training_state = c.get('training_state', 'ground_truth')
    if training_state not in ['ground_truth', 'mixed_recycling']:
        raise ValueError('Unknown training state policy')
    recycling_policies = {'high_noise_recycling': 'high_noise_stratified', 'uniform_recycling': 'uniform',
                         'high_noise_recycling_dual_head': 'high_noise_stratified',
                         **{k: 'high_noise_stratified' for k in [*reference_variants, *adapter_variants]}}
    registered_recycling = c.get('experiment_variant') in recycling_policies
    if training_state == 'mixed_recycling' or registered_recycling or 'recycle_fraction' in c:
        if not (registered_recycling and training_state == 'mixed_recycling'
                and c['phase'] == 'P1' and not c['debug'] and c['method'] == 'diffusion'
                and not c.get('fixed_crops') and policy == recycling_policies[c['experiment_variant']]
                and c['batch_size'] >= 2 and c['batch_size'] % 2 == 0
                and 0 < c.get('high_noise_start', 900) < c['timesteps']
                and c.get('recycle_fraction') == 0.5):
            raise ValueError('Recycling requires a registered P1 variant, matching timestep policy and fraction=0.5')
    if c.get('fixed_crops') and (not c['debug'] or c['phase'] != 'P0'):
        raise ValueError('Fixed training crops are restricted to the P0 debug diagnostic')
    if not isinstance(c.get('dual_head', False), bool):
        raise ValueError('dual_head must be boolean')
    if c.get('dual_head', False) or c.get('experiment_variant') == 'high_noise_recycling_dual_head':
        if not (c.get('dual_head') is True and c.get('experiment_variant') == 'high_noise_recycling_dual_head'
                and training_state == 'mixed_recycling' and policy == 'high_noise_stratified'
                and c['method'] == 'diffusion' and c['phase'] == 'P1' and not c['debug']):
            raise ValueError('Dual heads require the registered high-noise recycling P1 variant')


def train(config, project, run, resume=None, stop_after_epoch=None):
    validate_config(config)
    project, run = Path(project), Path(run)
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device(config['device'])
    seed_all(config['seed'])
    guard = StorageGuard(project, run, config['storage'])
    guard.check()
    manifest = read_manifest(config['manifest'])
    training = dataset_from_config(config, 'train', crop=config['crop_size'], seed=config['seed'],
                               limit=config['train_limit'])
    validation = dataset_from_config(config, 'val',
                                 crop=config['crop_size'] if config['debug'] else None,
                                 seed=config['seed'], augment=False, limit=config['val_limit'])
    if config.get('fixed_crops'):
        training = FixedCropDataset(config['manifest'], config['fixed_crops'])
        validation = FixedCropDataset(config['manifest'], config['fixed_crops'])
        if training.crop != config['crop_size']:
            raise ValueError('P0 config and crop specification disagree')
    model = make_model(config).to(device)
    adapter_mode = config.get('condition_architecture') == 'frozen_reference_adapter'
    parameters = sum(p.numel() for p in model.parameters())
    # Full per-run budget projection, including transient checkpoint, before expensive work.
    resident_budget = 16*parameters+(.25+.25+.03125)*1024**3
    guard.check(max(0, resident_budget-directory_size(run))+12*parameters+8*1024**2)
    diffusion = GaussianDiffusion(config['timesteps'], device)
    optim_parameters = [p for p in model.parameters() if p.requires_grad] if adapter_mode else model.parameters()
    optimizer = torch.optim.AdamW(optim_parameters, lr=config['lr'], weight_decay=config['weight_decay'])
    def lr_multiplier(epoch):
        warm = config['warmup_epochs']
        if epoch < warm:
            return (epoch+1)/max(warm, 1)
        progress = min(1., (epoch-warm)/max(1, config['epochs']-warm-1))
        minimum = config['min_lr']/config['lr']
        return minimum+(1-minimum)*.5*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    class_weights = None
    if config['class_balance']:
        counts = np.zeros(4, dtype=np.int64)
        # Only training labels contribute; all main training labels, even during debug.
        full_train = MaterialDataset(config['manifest'], 'train')
        for i in range(len(full_train)):
            counts += np.bincount(full_train.raw(i)[1].ravel(), minlength=4)
        weights = 1/np.sqrt(np.maximum(counts/counts.sum(), 1e-12))
        weights = np.clip(weights/weights.mean(), .25, 4.)
        class_weights = torch.tensor(weights, device=device, dtype=torch.float32)
    start_epoch, step, best = 0, 0, -float('inf')
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['config'] != config or 'optimizer' not in state:
            raise ValueError('Resume requires a full last checkpoint with identical config')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, step, best = state['epoch'], state['global_step'], state['best_miou']
        restore_rng(state['rng'])
        # A best file may have committed just before last was interrupted. It remains valid.
        best_path = run/'checkpoints/best.pt'
        if best_path.exists():
            previous_best = torch.load(best_path, map_location='cpu', weights_only=False)
            if previous_best['config'] != config:
                raise ValueError('Best checkpoint belongs to a different config')
            best = max(best, previous_best['validation_metrics']['miou_4class'])
    else:
        (run/'checkpoints').mkdir(parents=True)
        write_json(run/'config.json', config)
        write_json(run/'run_metadata.json', {
            'parameter_count': parameters, 'torch_version': torch.__version__,
            **({'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'warm_start_base_checkpoint': config['adapter_base_checkpoint'],
                'warm_start_base_sha256': config['adapter_base_sha256'],
                'training_budget': '40 epochs F0 base plus 40 epochs adapter; single training seed42',
                'all_rejected_batch_policy': 'loss logged; no backward or optimizer step; planned global_step advances'} if adapter_mode else {}),
            'python_version': platform.python_version(),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'split_hashes': {k: v['sha256'] for k, v in manifest['splits'].items()},
            'class_names': CLASSES, 'palette': PALETTE.tolist(),
            'preview_ids': validation.ids[:config['preview_count']],
            'class_weights': None if class_weights is None else class_weights.tolist(),
            'source_sha256': source_hashes(project), 'debug_only': config['debug'],
            'evaluation_data': 'fixed_train_crops' if config.get('fixed_crops') else 'val',
            'fixed_crops_sha256': hashlib.sha256(Path(config['fixed_crops']).read_bytes()).hexdigest() if config.get('fixed_crops') else None,
            'determinism': 'deterministic algorithms enforced; unsupported operations raise',
        })
    logger = ExperimentLogger(run, guard, step if resume else None, start_epoch if resume else None)
    if resume:
        write_json(run/f'resume_{time.time_ns()}.json', {'checkpoint': str(resume), 'epoch': start_epoch,
                   'global_step': step, 'source_sha256': source_hashes(project)})
    write_json(run/'status.json', {'status': 'running', 'last_completed_epoch': start_epoch})
    try:
        for epoch in range(start_epoch+1, config['epochs']+1):
            guard.check()
            training.epoch = epoch
            generator = torch.Generator().manual_seed(config['seed']+epoch)
            loader = DataLoader(training, batch_size=config['batch_size'], shuffle=True,
                                num_workers=config['workers'], generator=generator,
                                pin_memory=device.type == 'cuda', drop_last=False)
            model.train()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            begin, seen, sums = time.perf_counter(), 0, {}
            timestep_sum, high_noise_count = 0, 0
            reference_active, reference_confidence_sum = 0, 0.0
            adapter_updates, adapter_skipped, adapter_nonzero_updates = 0, 0, 0
            recycled_count, recycled_high_noise_count = 0, 0
            epoch_lr = optimizer.param_groups[0]['lr']
            for batch_index, (image, target, _) in enumerate(loader):
                if config['max_batches'] is not None and batch_index >= config['max_batches']:
                    break
                image, target = image.to(device), target.to(device)
                if config.get('condition_architecture') in ['split_rgb_state', 'frozen_reference_adapter']:
                    q = image[:, 6, 0, 0]
                    reference_active += int((q > 0).sum())
                    reference_confidence_sum += float(q.sum())
                optimizer.zero_grad(set_to_none=True)
                x0 = None
                if config['method'] == 'diffusion':
                    x0 = diffusion.encode(target)
                    t = diffusion.training_timesteps(image.shape[0],
                        config.get('timestep_sampling', 'uniform'), config.get('high_noise_start', 900))
                    timestep_sum += int(t.sum())
                    high_noise_count += int((t >= config.get('high_noise_start', 900)).sum())
                    noise = torch.randn_like(x0)
                    if config.get('training_state', 'ground_truth') == 'mixed_recycling':
                        noisy, selected = mixed_recycling_input(model, diffusion, image, x0, t, noise,
                                                                 config['recycle_fraction'])
                        recycled_count += selected.numel()
                        recycled_high_noise_count += int((t[selected] >= config['high_noise_start']).sum())
                    else:
                        noisy = diffusion.q_sample(x0, t, noise)
                    logits = model(image, noisy, t)
                else:
                    logits = model(image)
                loss, terms = losses(logits, target, config, x0, class_weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                if adapter_mode and not bool((q > 0).any()):
                    if loss.requires_grad:
                        raise RuntimeError('Rejected batch unexpectedly trains adapter')
                    grad = image.new_zeros(())
                    adapter_skipped += 1
                else:
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
                    optimizer.step()
                    if adapter_mode:
                        adapter_updates += 1
                        adapter_nonzero_updates += int(float(grad) > 0)
                step += 1
                n = image.shape[0]
                seen += n
                values = {k: float(v.detach()) for k, v in terms.items()}
                for k, v in values.items():
                    sums[k] = sums.get(k, 0.)+v*n
                if step % config['log_every'] == 0:
                    values.update(lr_group_0=epoch_lr, grad_norm_pre_clip=float(grad))
                    logger.scalars('steps', {'train_step/'+k: v for k, v in values.items()}, epoch, step)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            seconds = time.perf_counter()-begin
            if seen == 0:
                raise RuntimeError('No training samples')
            scalars = {'train_epoch/'+k: v/seen for k, v in sums.items()}
            if config.get('condition_architecture') in ['split_rgb_state', 'frozen_reference_adapter']:
                scalars.update({'reference_train/active_fraction': reference_active / seen,
                                'reference_train/mean_confidence': reference_confidence_sum / seen})
            if adapter_mode:
                scalars.update({'adapter_train/optimizer_updates': adapter_updates,
                                'adapter_train/nonzero_gradient_updates': adapter_nonzero_updates,
                                'adapter_train/all_rejected_batches': adapter_skipped,
                                'adapter_train/accepted_sample_exposures': reference_active})
            if config['method'] == 'diffusion':
                scalars.update({'diffusion_train/timestep_mean': timestep_sum/seen,
                                'diffusion_train/high_noise_fraction': high_noise_count/seen})
                if config.get('training_state') == 'mixed_recycling':
                    scalars.update({'diffusion_train/recycle_fraction': recycled_count/seen,
                                    'diffusion_train/recycled_high_noise_fraction': recycled_high_noise_count/max(1, recycled_count),
                                    'diffusion_train/teacher_sample_forwards': recycled_count,
                                    'diffusion_train/total_sample_forwards': seen+recycled_count})
            scalars.update({'resources/train_seconds': seconds, 'resources/train_samples_per_second': seen/seconds,
                            'optim/epoch_lr_group_0': epoch_lr})
            if device.type == 'cuda':
                scalars.update({'resources/train_peak_allocated_mib': torch.cuda.max_memory_allocated(device)/1024**2,
                                'resources/train_peak_reserved_mib': torch.cuda.max_memory_reserved(device)/1024**2})
            scalars.update({'resources/'+k: v for k, v in guard.check().items()})
            logger.scalars('epochs', scalars, epoch, step)
            val_metrics = None
            if epoch % config['val_every'] == 0 or epoch == config['epochs']:
                guard.check(16*1024**2)
                val_metrics, matrices, previews = evaluate(model, diffusion, validation, config)
                metric_prefix = 'overfit/' if config.get('fixed_crops') else 'val/'
                logger.scalars('val', {metric_prefix+k: v for k, v in val_metrics.items()}, epoch, step)
                if config['method'] == 'diffusion':
                    logger.scalars('val', {'sampling/steps': config['sampling_steps'],
                                          'sampling/samples_per_image': config['samples_per_image']}, epoch, step)
                logger.images(previews, epoch)
                np.savez_compressed(run/'latest_val_confusion.npz', **matrices)
                if val_metrics['miou_4class'] > best:
                    best = val_metrics['miou_4class']
                    atomic_checkpoint(run/'checkpoints/best.pt', {
                        'model': model.state_dict(), 'config': config, 'epoch': epoch,
                        'global_step': step, 'validation_metrics': val_metrics}, guard)
                    write_json(run/'best_val_metrics.json', {'epoch': epoch, 'metrics': val_metrics})
                    np.savez_compressed(run/'best_val_confusion.npz', **matrices)
            scheduler.step()
            logger.flush()
            atomic_checkpoint(run/'checkpoints/last.pt', {
                'model': model.state_dict(), 'config': config, 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'epoch': epoch,
                'global_step': step, 'best_miou': best}, guard)
            guard.check()
            write_json(run/'status.json', {'status': 'running', 'last_completed_epoch': epoch, 'global_step': step})
            print({'epoch': epoch, 'global_step': step, 'train_loss': sums['loss_total']/seen,
                   'val_miou': None if val_metrics is None else val_metrics['miou_4class'], 'run': str(run)}, flush=True)
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                break
        status = 'completed' if epoch == config['epochs'] else 'stopped_at_requested_epoch'
        write_json(run/'status.json', {'status': status, 'last_completed_epoch': epoch, 'global_step': step})
    except BaseException as exc:
        # Do not attempt another potentially large save after a storage or CUDA failure.
        try:
            write_json(run/'status.json', {'status': 'interrupted', 'error': str(exc), 'global_step': step,
                       'recovery': 'Use last successfully written checkpoints/last.pt'})
        except OSError:
            pass
        raise
    finally:
        logger.close()
    return run

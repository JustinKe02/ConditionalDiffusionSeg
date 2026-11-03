#!/usr/bin/env python3
"""Read-only CPU integration verification; print a machine-readable report.

No effect-training run is started. The
small synthetic optimizer exercises are exclusively restore-equivalence tests.
An additional read-only legacy Direct audit uses small true train/val crops,
without computing a new validation metric or changing saved research weights.
No report file is written: the launching task can preserve stdout in its batch.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

# Explicitly keep this verification process away from any concurrent GPU work.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))


def map_legacy_direct_weights(state, target_model):
    """Complete old Direct -> new S0 name mapping; never edits its inputs.

    The sole additional state is S0's fixed stage-grid buffer, unused in Direct.
    A missing/extra tensor or a shape/dtype mismatch is a hard error.
    """
    import torch
    if target_model.mode != 'S0':
        raise ValueError('Legacy Direct mapping is only defined for S0')
    prefixes = {'input.': 'rgb_encoder.input.', 'enc.': 'rgb_encoder.blocks.',
                'down.': 'rgb_encoder.down.', 'up.': 'decoder.up.',
                'dec.': 'decoder.blocks.', 'out.': 'head.'}
    target = target_model.state_dict()
    mapped = {'stage_timesteps': target['stage_timesteps'].clone()}
    for key, value in state.items():
        candidates = [new + key[len(old):] for old, new in prefixes.items() if key.startswith(old)]
        if len(candidates) != 1:
            raise ValueError('Unmapped or ambiguous legacy Direct key: ' + key)
        new_key = candidates[0]
        if new_key not in target or target[new_key].shape != value.shape or target[new_key].dtype != value.dtype:
            raise ValueError('Legacy Direct tensor mismatch: ' + key)
        mapped[new_key] = value.detach().clone()
    if mapped.keys() != target.keys():
        raise ValueError('Incomplete legacy Direct mapping')
    return mapped


def legacy_direct_reuse_audit():
    import torch
    from conditional_diffusion.model import SegUNet
    from conditional_diffusion.static_semantic import StaticSemanticModel
    from conditional_diffusion.static_experiment import scheduler_for, loader_for, configure
    from conditional_diffusion.data import MaterialDataset
    from conditional_diffusion.objectives import losses
    from static_semantic_worker import base_config
    old_run = ROOT / 'outputs/P1/direct/seed_42/p1_cosine_20261017'
    old_config = json.loads((old_run / 'config.json').read_text())
    candidate = base_config('S0', 2)
    common_keys = ['seed', 'manifest', 'device', 'cpu_threads', 'base_channels', 'iterations',
                   'crop_size', 'stride', 'batch_size', 'workers', 'epochs', 'warmup_epochs',
                   'lr', 'min_lr', 'weight_decay', 'grad_clip', 'class_balance', 'boundary',
                   'boundary_weight', 'boundary_metrics', 'boundary_tolerance', 'val_every',
                   'samples_per_image', 'preview_count', 'train_limit', 'val_limit', 'max_batches']
    config_values = {k: {'historical': old_config[k], 'candidate': candidate[k],
                         'equal': old_config[k] == candidate[k]} for k in common_keys}
    assert all(v['equal'] for v in config_values.values()), 'Legacy S0 common recipe differs'
    # Configure exactly the new platform, overriding only CPU execution for this check.
    # CUDA is hidden in this process; reading backend flags does not allocate a GPU.
    configure(dict(candidate, device='cpu', cpu_threads=2))
    backend_flags = {'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                     'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                     'historical_pytorch25_defaults': {'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': True}}
    assert backend_flags['cuda_matmul_allow_tf32'] is False and backend_flags['cudnn_allow_tf32'] is True, 'TF32 flags differ from historical FP32 defaults'
    mapping_prefix = {'input.': 'rgb_encoder.input.', 'enc.': 'rgb_encoder.blocks.',
                      'down.': 'rgb_encoder.down.', 'up.': 'decoder.up.',
                      'dec.': 'decoder.blocks.', 'out.': 'head.'}
    def translate(key):
        return next(new + key[len(old):] for old, new in mapping_prefix.items() if key.startswith(old))
    def compare_tensors(old, new, what):
        for key, value in old.items():
            assert torch.equal(value, new[translate(key)]), what + '/' + key
    updates = []
    for crop in [32, 64]:
        torch.manual_seed(42)
        old, new = SegUNet('direct', base=32), StaticSemanticModel('S0')
        compare_tensors(old.state_dict(), new.state_dict(), 'initial')
        old_optimizer = torch.optim.AdamW(old.parameters(), lr=2e-4, weight_decay=1e-4)
        new_optimizer = torch.optim.AdamW(new.parameters(), lr=2e-4, weight_decay=1e-4)
        old_scheduler = scheduler_for(old_optimizer, old_config)
        new_scheduler = scheduler_for(new_optimizer, candidate)
        dataset = MaterialDataset(candidate['manifest'], 'train', crop=crop, seed=42)
        c_cpu = dict(candidate, device='cpu', workers=0, cpu_threads=2)
        loader = loader_for(dataset, c_cpu, 1)
        seen = []
        for step, (image, target, ids) in enumerate(loader, 1):
            if step > 2:
                break
            old_optimizer.zero_grad(set_to_none=True); new_optimizer.zero_grad(set_to_none=True)
            old_output, new_output = old(image), new(image)
            assert torch.equal(old_output, new_output), 'Initial/trained forward differs'
            old_loss, _ = losses(old_output, target, old_config)
            new_loss, _ = losses(new_output, target, candidate)
            assert torch.equal(old_loss, new_loss), 'Loss differs'
            old_loss.backward(); new_loss.backward()
            compare_tensors({k:p.grad for k,p in old.named_parameters()},
                            {k:p.grad for k,p in new.named_parameters()}, 'preclip_gradient')
            old_norm = torch.nn.utils.clip_grad_norm_(old.parameters(), 1., error_if_nonfinite=True)
            new_norm = torch.nn.utils.clip_grad_norm_(new.parameters(), 1., error_if_nonfinite=True)
            assert torch.equal(old_norm, new_norm), 'Gradient norm differs'
            compare_tensors({k:p.grad for k,p in old.named_parameters()},
                            {k:p.grad for k,p in new.named_parameters()}, 'clipped_gradient')
            old_optimizer.step(); new_optimizer.step()
            compare_tensors(old.state_dict(), new.state_dict(), 'updated_parameters')
            for old_p, new_p in zip(old.parameters(), new.parameters()):
                for key, value in old_optimizer.state[old_p].items():
                    assert torch.equal(value, new_optimizer.state[new_p][key]), 'AdamW state differs'
            seen.extend(ids)
            updates.append({'crop': crop, 'update': step, 'ids': list(ids), 'loss': float(old_loss.detach()),
                            'output_gradient_parameter_adam_bitwise_equal': True})
        assert len(seen) == 4
        old_scheduler.step(); new_scheduler.step()
        assert old_scheduler.state_dict() == new_scheduler.state_dict()
    checkpoint_path = old_run / 'checkpoints/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    old = SegUNet('direct', base=32).eval()
    new = StaticSemanticModel('S0').eval()
    old.load_state_dict(checkpoint['model'])
    mapped = map_legacy_direct_weights(checkpoint['model'], new)
    new.load_state_dict(mapped, strict=True)
    small_validation = MaterialDataset(candidate['manifest'], 'val', crop=64, seed=42, augment=False)
    val_ids = []
    with torch.no_grad():
        for index in [0, 1]:
            image, _, ident = small_validation[index]
            assert torch.equal(old(image[None]), new(image[None])), 'Mapped historical best forward differs'
            val_ids.append(ident)
    return {'status': 'passed', 'mapping_tensor_count': len(checkpoint['model']),
            'extra_nonparameter_buffer': 'stage_timesteps; unused by S0',
            'shared_recipe': config_values, 'backend_flags': backend_flags,
            'direct_inert_difference': {'sampling_steps': [old_config['sampling_steps'], candidate['sampling_steps']],
                                        'reason': 'Direct executes one forward and uses no diffusion time grid.'},
            'real_train_updates': updates, 'validation_small_crop_ids': val_ids,
            'mapped_best_forward_bitwise_equal': True, 'validation_metrics_computed': False,
            'historical_best_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'reuse_conditions': 'Only if final common training batch is 2 and window batch is 1, full spatial/data/optimization/FP32/selection protocols stay identical. Changed batch requires a new S0 run.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-only', action='store_true', help='Only the new platform tests; default includes all legacy tests.')
    args = parser.parse_args()
    import torch
    from conditional_diffusion.static_semantic import StaticSemanticModel
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        raise RuntimeError('Verification must not access CUDA')
    source_paths = [ROOT / 'src/conditional_diffusion' / name for name in
                    ['static_semantic.py', 'static_experiment.py', 'model.py', 'diffusion.py', 'recycling.py', 'runtime.py']]
    source_paths += [ROOT / 'tests/test_static_semantic.py', Path(__file__).resolve()]
    def hashes():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    before = hashes()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'),
                pattern='test_static_semantic.py' if args.new_only else 'test_*.py')
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    legacy_audit = legacy_direct_reuse_audit() if result.wasSuccessful() else {'status': 'not_run_due_to_test_failure'}
    parameters = {}
    for mode in ['S0', 'S1', 'S2']:
        model = StaticSemanticModel(mode)
        parameters[mode] = {'total': sum(p.numel() for p in model.parameters()),
                            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    after = hashes()
    report = {'status': 'passed' if result.wasSuccessful() and before == after else 'failed',
              'scope': 'CPU 32/64 implementation checks; synthetic restore and matched real-train update tests; historical best mapping on fixed tiny val crops, no new metric, effect training, test data or GPU use.',
              'cpu_threads': torch.get_num_threads(), 'cuda_available': torch.cuda.is_available(),
              'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': len(result.skipped), 'all_legacy_tests_included': not args.new_only,
              'seconds': time.monotonic() - started, 'parameters': parameters,
              'legacy_direct_reuse_audit': legacy_audit,
              'sources_unchanged_during_verification': before == after, 'source_sha256': after,
              'limitations': 'Resume checks cover model/AdamW/scheduler/Python/NumPy/Torch CPU RNG; trainer I/O, CUDA numerical behavior, real batch throughput and full-image fusion require separate checks.'}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

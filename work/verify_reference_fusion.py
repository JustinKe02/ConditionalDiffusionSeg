"""Verify real old weights,512 resources, and exact new-path recovery before pilots."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import fcntl
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.model import make_model
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.objectives import losses
from conditional_diffusion.recycling import mixed_recycling_input
from conditional_diffusion.trainer import train, validate_config
from conditional_diffusion.runtime import seed_all, rng_state, restore_rng, write_json

BATCH = ROOT / 'outputs/launches/p1_reference_fusion_20261018'
BACKUP = ROOT / 'work/backups/p1_reference_fusion_20261018/src/conditional_diffusion'


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def verify_legacy():
    old_model = module('frozen_legacy_model', BACKUP / 'model.py')
    old_diffusion = module('frozen_legacy_diffusion', BACKUP / 'diffusion.py')
    old_recycling = module('frozen_legacy_recycling', BACKUP / 'recycling.py')
    results = []
    for method, batch in [('direct', 'p1_cosine_20261017'), ('iterative', 'p1_cosine_20261017'),
                           ('diffusion', 'p1_recycling_20261017')]:
        checkpoint = ROOT / 'outputs/P1' / method / 'seed_42' / batch / 'checkpoints/best.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        c = state['config']
        seed_all(42); a = old_model.make_model(c).cuda(); before = rng_state()
        seed_all(42); b = make_model(c).cuda(); assert equal(before, rng_state())
        a.load_state_dict(state['model']); b.load_state_dict(state['model'])
        image = torch.randn(2, 3, 32, 32, device='cuda')
        mask = torch.randint(4, (2, 32, 32), device='cuda')
        checks = dict(initialization_rng_equal=True, weights_loaded=True)
        with torch.no_grad():
            if method == 'diffusion':
                x0 = GaussianDiffusion.encode(mask)
                xt, t = torch.randn_like(x0), torch.tensor([100, 950], device='cuda')
                checks['forward_equal'] = torch.equal(a(image, xt, t), b(image, xt, t))
                da, db = old_diffusion.GaussianDiffusion(1000, 'cuda'), GaussianDiffusion(1000, 'cuda')
                noise = torch.randn_like(x0)
                checks['ddim_equal'] = torch.equal(da.sample(a, image, noise, 3), db.sample(b, image, noise, 3))
                old_loss = losses(a(image, xt, t), mask, c, x0)[0]
                checks['loss_equal'] = torch.equal(old_loss, losses(b(image, xt, t), mask, c, x0)[0])
                saved = rng_state()
                first = old_recycling.mixed_recycling_input(a, da, image, x0, t, noise)
                after = rng_state(); restore_rng(saved)
                second = mixed_recycling_input(b, db, image, x0, t, noise)
                checks['recycling_equal'] = equal(first, second)
                checks['recycling_rng_equal'] = equal(after, rng_state())
            else:
                checks['forward_equal'] = torch.equal(a(image), b(image))
        assert all(checks.values()), (method, checks)
        results.append(dict(method=method, checkpoint=str(checkpoint), checks=checks))
        del a, b, state
        torch.cuda.empty_cache()
    write_json(BATCH / 'legacy_verification.json', dict(status='passed', results=results,
               scope='Trusted actual old best weights and backed-up source; synthetic32 inputs; exact forward/loss/DDIM/recycling/RNG comparison. No old training, val or test evaluation.'))
    print('LEGACY_COMPATIBILITY_PASSED', flush=True)


def verify_resources():
    rows = []
    for variant in ['f0', 'f1']:
        c = json.loads((ROOT / f'configs/pilot_reference_{variant}.json').read_text())
        validate_config(c);seed_all(42)
        model = make_model(c).cuda().train()
        parameters = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        image = torch.randn(2, 7, 512, 512, device='cuda')
        if variant == 'f0':
            image[:, 3:] = 0
        else:
            image[:, 6] = .7
        mask = torch.randint(4, (2, 512, 512), device='cuda')
        diffusion = GaussianDiffusion(1000, 'cuda'); x0 = diffusion.encode(mask)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start = time.perf_counter()
        t = diffusion.training_timesteps(2, 'high_noise_stratified', 900)
        noisy, selected = mixed_recycling_input(model, diffusion, image, x0, t, torch.randn_like(x0))
        output = model(image, noisy, t)
        loss, _ = losses(output, mask, c, x0)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True);optimizer.step()
        torch.cuda.synchronize()
        rows.append(dict(variant=variant, parameters=parameters, seconds=time.perf_counter() - start,
                         peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2,
                         peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024 ** 2,
                         loss=float(loss.detach()), teacher_samples=selected.numel()))
        del loss, output, optimizer, model, image, mask, x0, noisy, diffusion
        torch.cuda.empty_cache()
    assert rows[0]['parameters'] == rows[1]['parameters']
    write_json(BATCH / 'resource_verification.json', dict(status='passed', results=rows,
               scope='One synthetic512/batch2/base32FP32 update per candidate including recycling teacher; not throughput or performance evidence.'))
    print('REFERENCE_RESOURCE_PASSED', rows, flush=True)


def verify_resume():
    rows = []
    for variant in ['f0', 'f1']:
        c = json.loads((ROOT / f'configs/pilot_reference_{variant}.json').read_text())
        c.update(base_channels=8, crop_size=32, stride=32, workers=0, preview_count=0, log_every=60)
        base = ROOT / 'outputs/integration' / (f'reference_{variant}_resume_' + str(time.time_ns()))
        full, resumed = base / 'full', base / 'resume'
        for run, stop in [(full, 2), (resumed, 1)]:
            run.mkdir(parents=True)
            train(dict(c), ROOT, run, stop_after_epoch=stop)
        train(dict(c), ROOT, resumed, resume=resumed / 'checkpoints/last.pt', stop_after_epoch=2)
        a, b = [torch.load(run / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
                for run in [full, resumed]]
        checks = {key: equal(a[key], b[key]) for key in ['model', 'optimizer', 'scheduler', 'rng']}
        assert all(checks.values()), (variant, checks)
        assert a['epoch'] == b['epoch'] == 2 and a['global_step'] == b['global_step'] == 120
        rows.append(dict(variant=variant, bitwise_equal=checks, global_step=120, runs=[str(full), str(resumed)]))
        write_json(BATCH / 'resume_verification_partial.json', dict(status='running' if variant == 'f0' else 'passed', results=rows))
        print('REFERENCE_RESUME_PASSED', variant, flush=True)
    write_json(BATCH / 'resume_verification.json', dict(status='passed', results=rows,
               scope='Registered new P1 paths, full120train/base8/crop32, controlled two epochs of original40-epoch schedule, no val/test.'))


def main():
    lock = (ROOT / 'outputs/.experiments.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    assert not (BATCH / 'resume_verification.json').exists(), 'Preserve existing preflight results'
    verify_legacy();verify_resources();verify_resume()


if __name__ == '__main__':
    main()

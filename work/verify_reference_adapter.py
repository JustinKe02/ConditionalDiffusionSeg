"""Real-weight compatibility, frozen-base invariance, resources and exact recovery."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import csv
import fcntl
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'work'))
from conditional_diffusion.model import make_model
from conditional_diffusion.reference_data import dataset_from_config
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.inference import predict_image
from conditional_diffusion.metrics_impl import confusion
from conditional_diffusion.objectives import losses
from conditional_diffusion.recycling import mixed_recycling_input
from conditional_diffusion.trainer import train, validate_config
from conditional_diffusion.runtime import seed_all, rng_state, write_json
from verify_reference_fusion import equal, verify_legacy
import verify_reference_fusion as previous

BATCH = ROOT / 'outputs/launches/p1_reference_adapter_20261018'
OLD = ROOT / 'outputs/launches/p1_reference_fusion_20261018'


def read(path):
    return json.loads(Path(path).read_text())


def config(variant):
    return read(ROOT / f'configs/pilot_reference_adapter_{variant}.json')


def frozen_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.base.state_dict().items()}


def resources():
    rows = []
    initial, after_rng = None, None
    for v in ['a', 'b']:
        c = config(v); validate_config(c); seed_all(42)
        model = make_model(c).cuda().train()
        if initial is None:
            initial = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
            after_rng = rng_state()
        else:
            assert equal(initial, {k: x.detach().cpu() for k, x in model.state_dict().items()})
            assert equal(after_rng, rng_state())
        old = frozen_state(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c['lr'], weight_decay=c['weight_decay'])
        image = torch.randn(2, 7, 512, 512, device='cuda'); image[:, 6] = .8
        mask = torch.randint(4, (2, 512, 512), device='cuda')
        d = GaussianDiffusion(1000, 'cuda'); x0 = d.encode(mask)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
        t = d.training_timesteps(2, 'high_noise_stratified', 900)
        xt, selected = mixed_recycling_input(model, d, image, x0, t, torch.randn_like(x0))
        output = model(image, xt, t); loss = losses(output, mask, c, x0)[0]
        loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize()
        assert equal(old, frozen_state(model))
        assert all(p.grad is None for p in model.base.parameters()) and float(grad) > 0
        allocated = torch.cuda.max_memory_allocated() / 1024**2
        seconds = time.perf_counter() - start
        image[:, 6] = 0
        assert not model(image, xt, t).requires_grad
        rows.append(dict(variant=v, parameters=sum(p.numel() for p in model.parameters()),
                         trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                         peak_allocated_mib=allocated, seconds=seconds, base_unchanged=True,
                         teacher_samples=selected.numel(), all_rejected_no_grad=True))
        del model, optimizer, image, mask, output, loss, xt, x0, d
        torch.cuda.empty_cache()
    assert rows[0]['parameters'] == rows[1]['parameters']
    write_json(BATCH / 'resource_verification.json', dict(status='passed', results=rows, matching_init_and_rng=True,
               scope='Actual frozen F0 weights; synthetic512 batch2 FP32, teacher and update, all-rejected path.'))
    print('ADAPTER_RESOURCES_PASSED', rows, flush=True)


def resume_check():
    rows = []
    for v in ['a', 'b']:
        c = config(v)
        c.update(crop_size=32, stride=32, workers=0, preview_count=0, log_every=60)
        folder = ROOT / 'outputs/integration' / ('reference_adapter_' + v + '_' + str(time.time_ns()))
        runs = [folder / 'full', folder / 'resume']
        for run, stop in zip(runs, [2, 1]):
            run.mkdir(parents=True); train(dict(c), ROOT, run, stop_after_epoch=stop)
        train(dict(c), ROOT, runs[1], resume=runs[1] / 'checkpoints/last.pt', stop_after_epoch=2)
        a, b = [torch.load(run / 'checkpoints/last.pt', map_location='cpu', weights_only=False) for run in runs]
        checks = {key: equal(a[key], b[key]) for key in ['model', 'optimizer', 'scheduler', 'rng']}
        assert all(checks.values()), (v, checks)
        assert a['global_step'] == b['global_step'] == 120 and a['epoch'] == b['epoch'] == 2
        original = torch.load(c['adapter_base_checkpoint'], map_location='cpu', weights_only=False)['model']
        assert all(torch.equal(value, a['model']['base.' + key]) for key, value in original.items())
        counts = {}
        with (runs[0] / 'metrics_epochs.csv').open() as f:
            for row in csv.DictReader(f):
                if row['tag'].startswith('adapter_train/'):
                    counts[row['tag']] = counts.get(row['tag'], 0) + float(row['value'])
        assert counts['adapter_train/accepted_sample_exposures'] == 142
        assert counts['adapter_train/optimizer_updates'] + counts['adapter_train/all_rejected_batches'] == 120
        assert counts['adapter_train/all_rejected_batches'] > 0
        rows.append(dict(variant=v, bitwise_equal=checks, base_unchanged=True, global_step=120,
                         counters=counts, runs=list(map(str, runs))))
        write_json(BATCH / 'resume_verification_partial.json', dict(status='running', results=rows))
        print('ADAPTER_RESUME_PASSED', v, flush=True)
    write_json(BATCH / 'resume_verification.json', dict(status='passed', results=rows,
               scope='Actual F0/base32, full120train/crop32, two controlled epochs of unchanged40 schedule. No validation/test.'))


class AuditedFallback(nn.Module):
    method = 'diffusion'
    iterations = 5

    def __init__(self, candidate, original):
        super().__init__(); self.candidate = candidate; self.original = original; self.calls = 0

    def forward(self, image, state, t):
        assert torch.count_nonzero(image[:, 6]) == 0
        out = self.candidate(image, state, t)
        baseline_image = torch.cat([image[:, :3], torch.zeros_like(image[:, 3:])], dim=1)
        expected = self.original(baseline_image, state, t)
        assert torch.equal(out, expected), 'Rejected image numerical fallback differs'
        self.calls += 1
        return out


def fallback_check():
    c = config('b'); seed_all(42)
    model = make_model(c).cuda().eval()
    # Load a genuinely adapted checkpoint from the successful resume check.
    trial = read(BATCH / 'resume_verification.json')['results'][1]['runs'][0]
    trained = torch.load(Path(trial) / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(trained['model'])
    saved = torch.load(c['adapter_base_checkpoint'], map_location='cpu', weights_only=False)
    original = make_model(saved['config']).cuda().eval(); original.load_state_dict(saved['model'])
    assert all(torch.equal(v, model.base.state_dict()[k].cpu()) for k, v in saved['model'].items())
    ds = dataset_from_config(c, 'val', crop=None, augment=False)
    cache = read(c['reference_cache'])['images']
    indices = [i for i, ident in enumerate(ds.ids) if not cache[ident]['accepted']]
    assert len(indices) == 10
    historical = read(OLD / 'comparison_summary.json')['results']['f0']
    rows = []
    for sampling_seed in [3407, 3408]:
        if sampling_seed == 3407:
            path = Path(c['adapter_base_checkpoint']).parent.parent / 'best_val_confusion.npz'
        else:
            path = Path(historical['3408']['evaluation_directory']) / 'confusion_matrices.npz'
        with np.load(path) as data:
            previous = {str(ident): matrix.copy() for ident, matrix in zip(data['ids'], data['per_image'])}
        for index in indices:
            image, target, ident = ds[index]
            audited = AuditedFallback(model, original).eval()
            begin = time.perf_counter()
            prediction, nfe = predict_image(audited, GaussianDiffusion(1000, 'cuda'), image, ident,
                                            dict(c, sampling_seed=sampling_seed))
            assert np.array_equal(confusion(target.numpy(), prediction), previous[ident])
            rows.append(dict(id=ident, sampling_seed=sampling_seed, nfe=nfe, exact_forward_comparisons=audited.calls,
                             original_confusion_equal=True, prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),
                             seconds=time.perf_counter() - begin))
            write_json(BATCH / 'fallback_verification_partial.json', dict(status='running', results=rows,
                       detail='Each DDIM step compared numerically against independent original F0; identical trajectory implies identical fused prediction.'))
            print('FALLBACK_IMAGE_PASSED', sampling_seed, ident, len(rows), '/20', flush=True)
    write_json(BATCH / 'fallback_verification.json', dict(status='passed', results=rows,
               scope='All10 rejected native val images,20DDIM,512/384,both sampling seeds; adapted branch; exact independent F0 outputs at every step, historical per-image matrices equal. No test.'))


def main():
    lock = (ROOT / 'outputs/.experiments.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    assert not (BATCH / 'preflight_completed.json').exists()
    previous.BATCH = BATCH
    verify_legacy(); resources(); resume_check(); fallback_check()
    write_json(BATCH / 'preflight_completed.json', dict(status='passed', training_not_yet_started=True))


if __name__ == '__main__':
    main()

"""Forward-only matched comparison on the original twelve diagnostic train crops."""
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.data import MaterialDataset
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.model import make_model
from conditional_diffusion.inference import image_seed
from conditional_diffusion.metrics_impl import confusion, summarize
from conditional_diffusion.runtime import seed_all, source_hashes


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', default='p1_high_noise_20261017',
                        help='Completed batch providing the frozen current-source map and output directory')
    parser.add_argument('--runs', nargs='+', default=['p1_cosine_20261017', 'p1_high_noise_20261017'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    seed_all(42)
    batch = ROOT / 'outputs/launches' / args.batch
    output = batch / 'analysis/matched_train_patch_followup.json'
    if output.exists():
        raise FileExistsError(f'Preserving existing diagnosis: {output}')
    frozen = json.loads((batch / 'plan.json').read_text())['source_sha256']
    assert source_hashes(ROOT) == frozen
    old_report = json.loads((ROOT / 'outputs/launches/p1_cosine_20261017/analysis/train_patch_diagnosis.json').read_text())
    base = ROOT / 'outputs/P1/diffusion/seed_42'
    config = json.loads((base / args.runs[-1] / 'config.json').read_text())
    ds = MaterialDataset(config['manifest'], 'train', crop=512, seed=42, augment=True)
    ds.epoch = 40
    indices = old_report['indices']
    patches = [ds[i] for i in indices]
    assert [p[2] for p in patches] == old_report['ids']
    report = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  scope='Forward-only, same 12 previously fixed training crops at epoch40/seed42; not full-train, val or test metrics. No optimizer or parameter updates.',
                  sampling_seed=3407, indices=indices, ids=old_report['ids'], methods={},
                  diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256=frozen)
    for name in args.runs:
        path = base / name / 'checkpoints/best.pt'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = make_model(state['config']).cuda().eval()
        model.load_state_dict(state['model'])
        diff = GaussianDiffusion(1000, 'cuda')
        grid = torch.linspace(999, 0, 20, device='cuda').round().long()
        rollout = [np.zeros((4, 4), dtype=np.int64) for _ in grid]
        teacher = {t: np.zeros((4, 4), dtype=np.int64) for t in [500, 999]}
        swapped = {t: np.zeros((4, 4), dtype=np.int64) for t in [500, 999]}
        clipped_changed = np.zeros(20)
        outside = np.zeros(20)
        for i, (image, target, ident) in enumerate(patches):
            image = image[None].cuda()
            mask = target[None].cuda()
            gen = torch.Generator().manual_seed(image_seed(ident, 3407))
            noise = torch.randn((1, 4, 512, 512), generator=gen).cuda()
            xt = noise.clone()
            for j, t in enumerate(grid):
                raw = model(image, xt, t.expand(1))
                clean = raw.clamp(-1, 1)
                rollout[j] += confusion(target.numpy(), clean.argmax(1)[0].cpu().numpy())
                clipped_changed[j] += float((raw.argmax(1) != clean.argmax(1)).float().mean()) / len(patches)
                outside[j] += float((raw.abs() > 1).float().mean()) / len(patches)
                if j < len(grid) - 1:
                    a, ap = diff.abar[t], diff.abar[grid[j + 1]]
                    eps = (xt - a.sqrt() * clean) / (1 - a).sqrt()
                    xt = ap.sqrt() * clean + (1 - ap).sqrt() * eps
            for tval in teacher:
                t = torch.tensor([tval], device='cuda')
                xt = diff.q_sample(diff.encode(mask), t, noise)
                pred = model(image, xt, t).argmax(1)[0].cpu().numpy()
                teacher[tval] += confusion(target.numpy(), pred)
                other_image = patches[(i + 1) % len(patches)][0][None].cuda()
                pred = model(other_image, xt, t).argmax(1)[0].cpu().numpy()
                swapped[tval] += confusion(target.numpy(), pred)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        item = dict(checkpoint=str(path), checkpoint_sha256=digest, epoch=state['epoch'],
                    rollout=[dict(t=int(t), metrics=summarize(rollout[j]),
                                  clip_changes_argmax_fraction=float(clipped_changed[j]),
                                  output_outside_target_range_fraction=float(outside[j])) for j, t in enumerate(grid)],
                    teacher_forced={str(t): summarize(m) for t, m in teacher.items()},
                    teacher_forced_permuted_rgb={str(t): summarize(m) for t, m in swapped.items()})
        report['methods'][name] = item
        print(json.dumps(dict(method=name, trajectory_miou=[x['metrics']['miou_4class'] for x in item['rollout']],
                              teacher={t:m['miou_4class'] for t,m in item['teacher_forced'].items()},
                              swapped_rgb={t:m['miou_4class'] for t,m in item['teacher_forced_permuted_rgb'].items()},
                              final_metrics=item['rollout'][-1]['metrics']), ensure_ascii=False), flush=True)
        del model, state
        torch.cuda.empty_cache()
    assert source_hashes(ROOT) == frozen
    report['frozen_source_verified_unchanged'] = True
    out = batch / 'analysis'
    out.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

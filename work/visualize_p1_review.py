"""Scientific previews from recorded evaluations plus one matched val-image replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import json
import csv
import hashlib
import fcntl
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from conditional_diffusion.data import MaterialDataset, PALETTE, MEAN, STD
from conditional_diffusion.model import make_model
from conditional_diffusion.diffusion import GaussianDiffusion
from conditional_diffusion.inference import predict_image
from conditional_diffusion.metrics_impl import confusion, summarize
from conditional_diffusion.runtime import source_hashes, seed_all, write_json, StorageGuard

OUT = ROOT / 'outputs/analysis/p1_visual_review_20261017'
REFERENCES = {
    'original20': 'outputs/launches/p1_high_noise_20261017/baseline_seed3408.json',
    'high20': 'outputs/launches/p1_high_noise_20261017/candidate_seed3408.json',
    'uniform_recycle20': 'outputs/launches/p1_uniform_recycling_20261017/candidate_seed3408.json',
    'combined20': 'outputs/launches/p1_recycling_20261017/candidate_seed3408.json',
    'combined5': 'outputs/launches/p1_uniform_recycling_20261017/recycling_steps5_seed3408.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legend(fig, error=False):
    names = ['Background', 'Monolayer', 'Few-layer', 'Multilayer']
    handles = [Patch(facecolor=color / 255, label=name) for color, name in zip(PALETTE, names)]
    if error:
        handles.append(Patch(facecolor='red', label='Wrong pixel (error panel)'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=10)


def save(fig, stem):
    fig.savefig(OUT / (stem + '.png'), dpi=170, facecolor='white')
    fig.savefig(OUT / (stem + '.pdf'), facecolor='white')
    plt.close(fig)
    print('FIGURE_SAVED', stem, flush=True)


def main():
    lock = open(ROOT / 'outputs/.experiments.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((ROOT / 'outputs/launches/p1_uniform_recycling_20261017/plan.json').read_text())
    assert source_hashes(ROOT) == plan['source_sha256']
    assert not OUT.exists(), 'Preserve previous visual-review artifacts'
    OUT.mkdir(parents=True)
    refs = {k: json.loads((ROOT / p).read_text()) for k, p in REFERENCES.items()}
    all_checkpoints = {x['checkpoint']: x['checkpoint_sha256'] for x in refs.values()}
    direct_run = ROOT / 'outputs/P1/direct/seed_42/p1_cosine_20261017'
    direct_path = direct_run / 'checkpoints/best.pt'
    all_checkpoints[str(direct_path)] = sha(direct_path)
    assert all(sha(p) == h for p, h in all_checkpoints.items())
    cfg = json.loads((ROOT / 'outputs/P1/diffusion/seed_42/p1_recycling_20261017/config.json').read_text())
    guard = StorageGuard(ROOT, OUT, cfg['storage'])
    guard.check(64 * 1024**2)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    cached = {}
    for key, ref in refs.items():
        assert ref['protocol']['sampling_seed'] == 3408
        cached[key] = {}
        for ident in ['m108', 'm109', 'm120', 'm127']:
            arr = np.asarray(Image.open(Path(ref['evaluation_directory']) / (ident + '_preview.png')))
            assert arr.shape == (384, 2048, 3)
            cached[key][ident] = [arr[:, j * 512:(j + 1) * 512] for j in range(4)]
            assert np.array_equal(cached[key][ident][1], cached['original20'][ident][1])
    # Existing fixed first-four previews, never selected for favorable results.
    fig, axes = plt.subplots(4, 5, figsize=(16, 10.7))
    titles = ['RGB input', 'Ground truth', 'Original diffusion (20)', 'High-noise + recycling (20)', 'Combined error map']
    for i, ident in enumerate(['m108', 'm109', 'm120', 'm127']):
        tiles = [cached['combined20'][ident][0], cached['combined20'][ident][1],
                 cached['original20'][ident][2], cached['combined20'][ident][2], cached['combined20'][ident][3]]
        for j, tile in enumerate(tiles):
            axes[i, j].imshow(tile, interpolation='nearest')
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel(ident, fontsize=12)
    fig.suptitle('Recorded validation examples | sampling seed 3408 | no test data', fontsize=14)
    fig.subplots_adjust(left=.04, right=.995, top=.94, bottom=.06, wspace=.025, hspace=.06)
    legend(fig, error=True)
    save(fig, 'recorded_validation_examples')

    # A readable six-panel overview of the known difficult, fixed preview m109.
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.9))
    ident = 'm109'
    tiles = [('RGB input', cached['combined20'][ident][0]), ('Ground truth', cached['combined20'][ident][1]),
             ('Original diffusion (20)', cached['original20'][ident][2]),
             ('High-noise only (20)', cached['high20'][ident][2]),
             ('Uniform + recycling (20)', cached['uniform_recycle20'][ident][2]),
             ('High-noise + recycling (20)', cached['combined20'][ident][2])]
    for ax, (title, tile) in zip(axes.ravel(), tiles):
        ax.imshow(tile, interpolation='nearest'); ax.axis('off'); ax.set_title(title)
    fig.suptitle('m109 | fixed validation preview with broad class errors | seed 3408', fontsize=13)
    fig.subplots_adjust(top=.9, bottom=.07, left=.01, right=.99, wspace=.02, hspace=.1)
    legend(fig)
    save(fig, 'm109_ablation_comparison')

    # Training curves and measured full-val inference budget.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    entries = [('direct', 'direct', 'p1_cosine_20261017', '#444444'),
               ('Uniform', 'diffusion', 'p1_cosine_20261017', '#999999'),
               ('High noise', 'diffusion', 'p1_high_noise_20261017', '#D48A20'),
               ('Uniform + recycling', 'diffusion', 'p1_uniform_recycling_20261017', '#2B78A0'),
               ('High noise + recycling', 'diffusion', 'p1_recycling_20261017', '#A33160')]
    for label, method, run_id, color in entries:
        run = ROOT / 'outputs/P1' / method / 'seed_42' / run_id
        rows = list(csv.DictReader((run / 'metrics_val.csv').open()))
        points = [r for r in rows if r['tag'] == 'val/miou_4class']
        axes[0].plot([int(r['epoch']) for r in points], [float(r['value']) for r in points],
                     marker='o', color=color, label=label)
    axes[0].set(xlabel='Epoch', ylabel='Full-val four-class mIoU', xticks=[10, 20, 30, 40], ylim=(.5, .9),
                title='One training seed (42); sampling seed 3407')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1)); axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    sweep = json.loads((ROOT / 'outputs/launches/p1_uniform_recycling_20261017/recycling_step_sweep.json').read_text())
    for seed, color in [(3407, '#A33160'), (3408, '#2B78A0')]:
        vals = [sweep[f'recycling_steps{n}_seed{seed}']['metrics'] for n in [1, 5, 20]]
        axes[1].plot([v['seconds'] / 60 for v in vals], [v['miou_4class'] for v in vals],
                     'o-', color=color, label=f'Combined, seed {seed}')
        for steps, v in zip([1, 5, 20], vals):
            if seed == 3407:
                axes[1].annotate(f'{steps} steps', (v['seconds'] / 60, v['miou_4class']),
                                 xytext=(5, 7), textcoords='offset points', fontsize=9)
    for method, marker in [('direct', 's'), ('iterative', '^')]:
        m = json.loads((ROOT / 'outputs/P1' / method / 'seed_42/p1_cosine_20261017/best_val_metrics.json').read_text())['metrics']
        axes[1].scatter(m['seconds'] / 60, m['miou_4class'], marker=marker, s=55, label=method)
    axes[1].set(xlabel='Full 25-image validation time (minutes)', ylabel='Four-class mIoU',
                title='Measured inference cost; fixed weights', ylim=(.78, .85))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1)); axes[1].grid(alpha=.2); axes[1].legend(fontsize=8)
    save(fig, 'training_and_inference_cost')

    # Native-resolution replay of one of the already-fixed first-four val examples.
    torch.set_num_threads(4); seed_all(42)
    ds = MaterialDataset(cfg['manifest'], 'val', augment=False)
    image, target, ident = ds[ds.ids.index('m109')]
    target = target.numpy()
    rgb, _, _ = ds.raw(ds.ids.index(ident))
    predictions, replay = {}, {}
    tasks = [('direct', direct_path, 1, None),
             ('combined5', Path(refs['combined5']['checkpoint']), 5, refs['combined5']),
             ('combined20', Path(refs['combined20']['checkpoint']), 20, refs['combined20'])]
    for label, checkpoint, steps, reference in tasks:
        print('NATIVE_REPLAY_START', label, ident, flush=True)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = dict(state['config'], sampling_steps=steps, sampling_seed=3408, samples_per_image=1)
        model = make_model(config).cuda().eval(); model.load_state_dict(state['model'])
        pred, nfe = predict_image(model, GaussianDiffusion(config['timesteps'], 'cuda'), image, ident, config)
        matrix = confusion(target, pred)
        reference_path = direct_run / 'best_val_confusion.npz' if reference is None else Path(reference['evaluation_directory']) / 'confusion_matrices.npz'
        recorded = np.load(reference_path)
        expected = recorded['per_image'][list(recorded['ids']).index(ident)]
        assert np.array_equal(matrix, expected), f'Replay does not match recorded confusion: {label}'
        predictions[label] = pred
        guard.check(16 * 1024**2)
        Image.fromarray(pred).save(OUT / f'{ident}_{label}_labels.png')
        # Boundary band: native 17x17 neighborhood contains multiple GT classes.
        mask = torch.from_numpy(target.astype(np.float32))[None, None]
        padded = F.pad(mask, (8, 8, 8, 8), mode='replicate')
        band = (F.max_pool2d(padded, 17, 1) != -F.max_pool2d(-padded, 17, 1))[0, 0].numpy()
        wrong = pred != target
        replay[label] = dict(metrics=summarize(matrix), nfe=nfe, reference_confusion_exact_match=True,
                             checkpoint_sha256=sha(checkpoint),
                             wrong_pixels=int(wrong.sum()),
                             error_fraction_outside_gt_boundary_band=float((wrong & ~band).sum() / max(1, wrong.sum())),
                             boundary_band_definition='17x17 GT neighborhood contains multiple classes (~8px radius)')
        print('NATIVE_REPLAY_DONE', label, json.dumps(replay[label]), flush=True)
        del model, state
        torch.cuda.empty_cache()

    # Pick a native 512 ROI with maximal semantic error away from GT boundaries.
    wrong = ((predictions['combined20'] != target) & ~band).astype(np.int64)
    integral = np.pad(wrong.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    candidates = [(int(integral[y+512, x+512] - integral[y, x+512] - integral[y+512, x] + integral[y, x]), y, x)
                  for y in range(0, target.shape[0] - 512 + 1, 128)
                  for x in range(0, target.shape[1] - 512 + 1, 128)]
    _, top, left = max(candidates)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.6))
    arrays = [rgb, PALETTE[target], PALETTE[predictions['direct']], PALETTE[predictions['combined5']], PALETTE[predictions['combined20']]]
    titles = ['RGB input', 'Ground truth', 'Direct', 'Diffusion combined (5)', 'Diffusion combined (20)']
    for j, (arr, title) in enumerate(zip(arrays, titles)):
        axes[0, j].imshow(arr, interpolation='nearest'); axes[0, j].set_title(title)
        axes[0, j].add_patch(Rectangle((left, top), 512, 512, fill=False, edgecolor='#49E6B4', linewidth=1.8))
        axes[1, j].imshow(arr[top:top+512, left:left+512], interpolation='nearest')
        for ax in axes[:, j]: ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel('Full image'); axes[1, 0].set_ylabel('Native 512 x 512 ROI')
    fig.suptitle('m109 | matched recorded validation predictions | seed 3408 | same ROI in every column', fontsize=13)
    fig.subplots_adjust(left=.04, right=.995, top=.91, bottom=.07, wspace=.025, hspace=.08)
    legend(fig)
    save(fig, 'm109_native_comparison')
    for label, arr in [('rgb', rgb), ('gt', PALETTE[target]), ('direct', PALETTE[predictions['direct']]),
                       ('diffusion5', PALETTE[predictions['combined5']]), ('diffusion20', PALETTE[predictions['combined20']])]:
        Image.fromarray(arr[top:top+512, left:left+512]).save(OUT / f'm109_roi_{label}.png')
    assert source_hashes(ROOT) == plan['source_sha256']
    assert all(sha(p) == h for p, h in all_checkpoints.items())
    write_json(OUT / 'provenance.json', dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               scope='Existing fixed first-four val previews plus one native m109 forward-only replay. No training, no test evaluation.',
               preview_ids=['m108', 'm109', 'm120', 'm127'], references=REFERENCES,
               checkpoints_sha256=all_checkpoints, source_sha256=plan['source_sha256'],
               native_replay=replay, roi=dict(id='m109', top=top, left=left, size=512,
               selection='Maximum20-step error count outside GT boundary band on128-stride512 grid; visualization only, no model selection'),
               note='Preview masks downscaled with nearest-neighbor; metrics use original resolution, never recomputed from display thumbnails.'))
    print('VISUAL_REVIEW_COMPLETE', OUT, flush=True)


if __name__ == '__main__':
    main()

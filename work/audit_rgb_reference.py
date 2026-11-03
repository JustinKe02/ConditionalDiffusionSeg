"""Train-only identifiability audit of an RGB-only robust spatial color reference."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/launches/p1_reference_fusion_20261018'
SETTINGS = dict(grid=64, polynomial_degree=2, irls_iterations=12, tukey_cutoff=4.685,
                scale_floor=2 / 255, fit_support_min=0.60, rgb_confidence_min=0.45,
                residual_scale=0.04, minimum_reference_level=0.02)
GATE = dict(minimum_accepted_image_fraction=0.50,
            maximum_accepted_background_reference_bias_p90=0.03,
            maximum_accepted_insufficient_background_fraction=0.05,
            minimum_background_fraction_for_audit=0.05,
            maximum_median_reference_bias_to_minority_contrast_ratio=0.50,
            minimum_minority_grid_pixels=8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rgb_reference_fit(rgb):
    """RGB-only interface: no labels, split identity, or image ID are available."""
    h, w = rgb.shape[:2]
    ys = np.linspace(0, h - 1, SETTINGS['grid']).round().astype(int)
    xs = np.linspace(0, w - 1, SETTINGS['grid']).round().astype(int)
    yy, xx = np.meshgrid(2 * ys / (h - 1) - 1, 2 * xs / (w - 1) - 1, indexing='ij')
    a = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], axis=-1).reshape(-1, 6)
    values = rgb[ys[:, None], xs[None, :]].reshape(-1, 3).astype(np.float64) / 255
    beta = np.zeros((6, 3))
    beta[0] = np.median(values, axis=0)
    weights = np.ones(len(a))
    for _ in range(SETTINGS['irls_iterations']):
        residual = values - a @ beta
        scale = np.maximum(SETTINGS['scale_floor'],
                           1.4826 * np.median(np.abs(residual - np.median(residual, axis=0)), axis=0))
        distance = np.sqrt(np.mean((residual / scale) ** 2, axis=1))
        u = distance / SETTINGS['tukey_cutoff']
        weights = np.maximum(0, 1 - u * u) ** 2
        weights[u >= 1] = 0
        if (weights > 0.05).sum() < 12:
            return dict(accepted=False, confidence=0.0, support=0.0, beta=beta,
                        a=a, values=values, ys=ys, xs=xs, predicted=np.clip(a @ beta, 0, 1))
        root_w = np.sqrt(weights)[:, None]
        beta = np.linalg.lstsq(a * root_w, values * root_w, rcond=None)[0]
    predicted = np.clip(a @ beta, 0, 1)
    support = float(np.mean(weights >= 0.5))
    inlier = weights >= 0.5
    median_residual = float(np.median(np.mean(np.abs(values[inlier] - predicted[inlier]), axis=1)))
    confidence = float(support * np.exp(-median_residual / SETTINGS['residual_scale']))
    accepted = (support >= SETTINGS['fit_support_min'] and confidence >= SETTINGS['rgb_confidence_min']
                and float(predicted.min()) >= SETTINGS['minimum_reference_level'])
    return dict(accepted=bool(accepted), confidence=confidence, support=support, beta=beta,
                a=a, values=values, ys=ys, xs=xs, predicted=predicted)


def main():
    manifest_path = ROOT / 'data_manifests/source_dataset.json'
    manifest = json.loads(manifest_path.read_text())
    split = manifest['splits']['train']
    assert sha(split['snapshot_path']) == split['sha256']
    ids = Path(split['snapshot_path']).read_text().split()
    assert len(ids) == 120
    plan = dict(estimator=SETTINGS, preregistered_gate=GATE, code_sha256=sha(__file__),
                train_ids=ids, train_split_sha256=split['sha256'], manifest_sha256=sha(manifest_path),
                scope='Fit from RGB only, then assess against train labels. Deterministic64x64 full-image grid; not full-pixel train performance. No val/test images or labels read.',
                confidence='RGB fit support/residual only; not a calibrated background probability.',
                gate_note='Exploratory screening thresholds registered before reading diagnostic labels; not statistical significance or physical calibration.')
    pp = OUT / 'reference_audit_plan.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == plan
    else:
        write(pp, plan)
    assert not (OUT / 'reference_audit.json').exists(), 'Preserve prior diagnosis; no silent rerun'
    rows, inputs = [], {}
    for number, ident in enumerate(ids, 1):
        image_path = Path(manifest['dataset_root']) / manifest['image_pattern'].format(id=ident)
        with Image.open(image_path) as im:
            rgb = np.asarray(im.convert('RGB')).copy()
        # Complete estimation before loading the label used solely for the audit.
        fitted = rgb_reference_fit(rgb)
        mask_path = Path(manifest['dataset_root']) / manifest['mask_pattern'].format(id=ident)
        with Image.open(mask_path) as im:
            target = np.asarray(im)
        labels = target[fitted['ys'][:, None], fitted['xs'][None, :]].ravel()
        bg = labels == 0
        row = dict(id=ident, accepted=fitted['accepted'], rgb_confidence=fitted['confidence'],
                   rgb_fit_support=fitted['support'], grid_background_fraction=float(bg.mean()),
                   full_background_fraction=float((target == 0).mean()),
                   background_reference_bias=None, background_rgb_mae=None,
                   minority_contrasts={}, bias_to_minority_contrast=None,
                   coefficient_rgb=fitted['beta'].tolist())
        if bg.sum() >= 12:
            # Label-informed reference exists only as an audit oracle; never an inference input.
            oracle_beta = np.linalg.lstsq(fitted['a'][bg], fitted['values'][bg], rcond=None)[0]
            oracle = np.clip(fitted['a'] @ oracle_beta, 0, 1)
            bias = float(np.sqrt(np.mean((fitted['predicted'][bg] - oracle[bg]) ** 2)))
            row['background_reference_bias'] = bias
            row['background_rgb_mae'] = float(np.mean(np.abs(fitted['values'][bg] - fitted['predicted'][bg])))
            contrasts = []
            for label, name in [(1, 'monolayer'), (2, 'fewlayer')]:
                region = labels == label
                if region.sum() >= GATE['minimum_minority_grid_pixels']:
                    contrast = float(np.sqrt(np.mean((fitted['values'][region] - oracle[region]) ** 2)))
                    row['minority_contrasts'][name] = contrast
                    contrasts.append(contrast)
            if contrasts:
                row['bias_to_minority_contrast'] = bias / max(min(contrasts), 1 / 255)
        rows.append(row)
        inputs[str(image_path)], inputs[str(mask_path)] = sha(image_path), sha(mask_path)
        if number % 20 == 0:
            print(f'REFERENCE_AUDIT {number}/120', flush=True)
    accepted = [r for r in rows if r['accepted']]
    biases = [r['background_reference_bias'] for r in accepted if r['background_reference_bias'] is not None]
    ratios = [r['bias_to_minority_contrast'] for r in accepted if r['bias_to_minority_contrast'] is not None]
    insufficient = [r for r in accepted if r['full_background_fraction'] < GATE['minimum_background_fraction_for_audit']]
    stats = dict(accepted_count=len(accepted), image_count=120, accepted_fraction=len(accepted) / 120,
                 accepted_insufficient_background_count=len(insufficient),
                 accepted_insufficient_background_fraction=len(insufficient) / max(1, len(accepted)),
                 accepted_bias_p90=None if not biases else float(np.quantile(biases, 0.9)),
                 accepted_bias_median=None if not biases else float(np.median(biases)),
                 accepted_minority_ratio_count=len(ratios),
                 accepted_minority_ratio_median=None if not ratios else float(np.median(ratios)))
    checks = dict(coverage=stats['accepted_fraction'] >= GATE['minimum_accepted_image_fraction'],
                  reference_bias=stats['accepted_bias_p90'] is not None and stats['accepted_bias_p90'] <= GATE['maximum_accepted_background_reference_bias_p90'],
                  no_frequent_false_acceptance=stats['accepted_insufficient_background_fraction'] <= GATE['maximum_accepted_insufficient_background_fraction'],
                  minority_resolution=bool(ratios) and stats['accepted_minority_ratio_median'] <= GATE['maximum_median_reference_bias_to_minority_contrast_ratio'])
    passed = all(checks.values())
    result = dict(status='completed', passed=passed, checks=checks, statistics=stats,
                  rows=rows, input_sha256=inputs, plan_sha256=sha(pp),
                  interpretation='RGB-only reference merits controlled validation pilots; this does not establish innovation or held-out benefit.' if passed else
                  'Registered RGB-only robust-reference screening failed. Stop this reference candidate before model training; preserve failure and do not loosen gate to pass.',
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    write(OUT / 'reference_audit.json', result)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    q = [r['rgb_confidence'] for r in rows]
    err = [np.nan if r['background_reference_bias'] is None else r['background_reference_bias'] for r in rows]
    colors = [r['full_background_fraction'] for r in rows]
    plot = axes[0].scatter(q, err, c=colors, vmin=0, vmax=1, cmap='viridis', s=22)
    axes[0].axhline(GATE['maximum_accepted_background_reference_bias_p90'], color='red', linestyle='--')
    axes[0].set(xlabel='RGB-only fit confidence', ylabel='Reference bias (RGB scale0-1)', title='Train images: color=background fraction')
    fig.colorbar(plot, ax=axes[0])
    axes[1].hist(biases, bins=20);axes[1].set(xlabel='Reference bias', ylabel='Accepted image count', title='Accepted image reference error')
    axes[2].hist(np.minimum(ratios, 3), bins=20);axes[2].axvline(.5, color='red', linestyle='--')
    axes[2].set(xlabel='Bias / minority contrast (display clipped at3)', ylabel='Accepted image count', title='Reference error vs minority signal')
    fig.suptitle('Train-only RGB reference identifiability audit; no segmentation training')
    fig.tight_layout();fig.savefig(OUT / 'reference_audit.png', dpi=180);plt.close(fig)
    write(OUT / 'status.json', dict(status='audit_passed' if passed else 'stopped',
          stage='reference_supported_ready_for_controlled_pilots' if passed else 'reference_identifiability_failed',
          updated_at_utc=datetime.now(timezone.utc).isoformat(), reference_checks=checks, reference_statistics=stats))
    print(json.dumps(dict(passed=passed, checks=checks, statistics=stats), indent=2), flush=True)


if __name__ == '__main__':
    main()

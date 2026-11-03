"""CPU-only reanalysis of saved test counts; never loads a model or changes runs."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/analysis/chapter_readiness_20261031'


def miou(m):
    tp = np.diagonal(m, axis1=-2, axis2=-1)
    union = m.sum(-1) + m.sum(-2) - tp
    v = np.divide(tp, union, out=np.full(tp.shape, np.nan), where=union > 0)
    return np.nanmean(v, axis=-1)


def boundary(b):
    p = np.divide(b[..., 0], b[..., 1], out=np.zeros(b.shape[:-1]), where=b[..., 1] > 0)
    r = np.divide(b[..., 2], b[..., 3], out=np.zeros(b.shape[:-1]), where=b[..., 3] > 0)
    f = np.divide(2*p*r, p+r, out=np.zeros_like(p), where=p+r > 0)
    f = np.where(b[..., 1] + b[..., 3] > 0, f, np.nan)
    return np.nanmean(f[..., 1:], axis=-1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources, data = {}, {}
    for folder in ['glpflow_test_20261022', 'baselines_test_20261022']:
        for path in sorted((ROOT/'outputs/inference'/folder).glob('*_confusion.npz')):
            name = path.stem.removesuffix('_confusion')
            with np.load(path) as z:
                d = {k: z[k].copy() for k in z.files}
            assert np.array_equal(d['global_matrix'], d['per_image'].sum(0)), name
            assert np.array_equal(d['boundary_counts'], d['per_image_boundary'].sum(0)), name
            data[name] = d
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    base = data['E3R']
    n = len(base['ids'])
    for name, d in data.items():
        assert np.array_equal(base['ids'], d['ids']), name
        assert np.array_equal(base['per_image'].sum(-1), d['per_image'].sum(-1)), name
    rng = np.random.default_rng(20261031)
    indices = rng.integers(0, n, size=(10000, n))
    result = {
        'scope': 'Post-hoc paired image bootstrap of saved test counts, not retraining or model selection.',
        'limitations': [
            'Intervals assume independent images; sample/parent-image independence is unverified.',
            'Conditional on this fixed checkpoint and split; does not estimate training-seed variation.',
            'Test data was historically exposed; this is descriptive sensitivity, not a new blind test.',
            'Percentile intervals with 10000 paired resamples; no multiplicity adjustment or confirmatory claims.'
        ],
        'resampling_seed': 20261031, 'resamples': len(indices), 'images': n,
        'metrics_percent': {}, 'paired_difference_pp': {}, 'sha256': sources,
    }
    for name, d in data.items():
        result['metrics_percent'][name] = {
            'miou': float(100*miou(d['global_matrix'])),
            'foreground_boundary_f1': float(100*boundary(d['boundary_counts'])),
        }
    a_iou = miou(base['per_image'][indices].sum(1))
    a_bf = boundary(base['per_image_boundary'][indices].sum(1))
    for name in ['E0', 'E1', 'E3', 'segformer_b1']:
        d = data[name]
        diffs = {
            'miou': 100*(a_iou-miou(d['per_image'][indices].sum(1))),
            'foreground_boundary_f1': 100*(a_bf-boundary(d['per_image_boundary'][indices].sum(1))),
        }
        result['paired_difference_pp'][f'E3R_minus_{name}'] = {
            key: {
                'point': result['metrics_percent']['E3R'][key]-result['metrics_percent'][name][key],
                'percentile_95_interval': np.quantile(delta, [.025, .975]).tolist(),
            } for key, delta in diffs.items()
        }
    (OUT/'saved_count_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

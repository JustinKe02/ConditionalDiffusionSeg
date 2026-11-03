"""Read existing artifacts and produce a chapter-level evidence inventory; no model execution."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/analysis/chapter3_evidence_audit_20261018'
MODELS = [
    ('D', 'Direct', 'direct', 'p1_cosine_20261017'),
    ('I', 'Iterative5', 'iterative', 'p1_cosine_20261017'),
    ('U', 'Diffusion: uniform', 'diffusion', 'p1_cosine_20261017'),
    ('H', 'Diffusion: high noise', 'diffusion', 'p1_high_noise_20261017'),
    ('R', 'Diffusion: uniform + recycling', 'diffusion', 'p1_uniform_recycling_20261017'),
    ('HR', 'Diffusion: high noise + recycling', 'diffusion', 'p1_recycling_20261017'),
    ('HR2', 'Diffusion: dual head', 'diffusion', 'p1_dual_head_20261018'),
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tag_sum(path, tag):
    with Path(path).open() as f:
        values = [float(r['value']) for r in csv.DictReader(f) if r['tag'] == tag]
    assert values
    return sum(values)


def main():
    if OUT.exists():
        assert read(OUT / 'experiment_inventory.json')['scope'] == 'Artifact-only inventory; no training/inference/test evaluation.'
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = read(ROOT / 'outputs/launches/p1_test_20261018/plan.json')
    current = {str(p.relative_to(ROOT)): sha(p) for folder in ['src', 'scripts', 'configs']
               for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in {'.py', '.json', '.sh'}}
    assert current == frozen['evaluation_source_sha256']
    inputs = {}
    rows, models = [], []
    for ident, model_name, method, batch in MODELS:
        run = ROOT / 'outputs/P1' / method / 'seed_42' / batch
        state = read(run / 'status.json')
        assert state['status'] == 'completed' and state['last_completed_epoch'] == 40 and state['global_step'] == 2400
        config, metadata, best = [read(run / name) for name in ['config.json', 'run_metadata.json', 'best_val_metrics.json']]
        assert config['seed'] == 42 and config['epochs'] == 40 and config['batch_size'] == 2
        evaluations = []
        for directory in sorted((run / 'evaluations').glob('*')):
            if not (directory / 'metrics.json').is_file():
                continue
            ec, metrics = read(directory / 'evaluation_config.json'), read(directory / 'metrics.json')
            c = ec['config']
            evaluations.append(dict(directory=str(directory), split=ec['split'], checkpoint_epoch=ec['checkpoint_epoch'],
                                    sampling_steps=c['sampling_steps'], sampling_seed=c['sampling_seed'],
                                    samples_per_image=c['samples_per_image'], metrics=metrics))
            for name in ['evaluation_config.json', 'metrics.json']:
                inputs[str(directory / name)] = sha(directory / name)
        secondary = [e for e in evaluations if e['split'] == 'val' and e['sampling_steps'] == 20
                     and e['sampling_seed'] == 3408 and e['samples_per_image'] == 1]
        tests = [e for e in evaluations if e['split'] == 'test']
        assert len(tests) <= 1 and (method != 'diffusion' or len(secondary) == 1)
        train_s = tag_sum(run / 'metrics_epochs.csv', 'resources/train_seconds')
        val_s = tag_sum(run / 'metrics_val.csv', 'val/seconds')
        models.append(dict(id=ident, name=model_name, method=method, run=str(run), config=config,
                           training_seed=42, independent_training_replicates=1, epochs=40, optimizer_updates=2400,
                           best_epoch=best['epoch'], validation_seed3407=best['metrics'],
                           validation_seed3408=None if not secondary else secondary[0]['metrics'],
                           test=None if not tests else tests[0], parameters=metadata['parameter_count'],
                           train_seconds_sum=train_s, training_validation_seconds_sum=val_s,
                           evaluations=evaluations, trained_source_sha256=metadata['source_sha256'],
                           best_checkpoint_sha256=sha(run / 'checkpoints/best.pt')))
        metrics = best['metrics']
        rows.append(dict(id=ident, name=model_name, training_seed=42, epochs=40, optimizer_updates=2400,
                         best_epoch=best['epoch'], parameters=metadata['parameter_count'],
                         val_miou_3407_percent=metrics['miou_4class'] * 100,
                         val_miou_3408_percent='' if not secondary else secondary[0]['metrics']['miou_4class'] * 100,
                         val_monolayer_iou_percent=metrics['iou_monolayer'] * 100,
                         val_fewlayer_iou_percent=metrics['iou_fewlayer'] * 100,
                         val_foreground_boundary_f1_percent=metrics['boundary_f1_foreground'] * 100,
                         test_miou_percent='' if not tests else tests[0]['metrics']['miou_4class'] * 100,
                         test_foreground_boundary_f1_percent='' if not tests else tests[0]['metrics']['boundary_f1_foreground'] * 100,
                         train_minutes=train_s / 60, training_validation_minutes=val_s / 60, run=str(run)))
        for name in ['status.json', 'config.json', 'run_metadata.json', 'best_val_metrics.json', 'metrics_epochs.csv', 'metrics_val.csv']:
            inputs[str(run / name)] = sha(run / name)
    audit_path = ROOT / 'data_manifests/full_audit.json'
    dataset = read(audit_path)
    for value in dataset['splits'].values():
        total = sum(value['pixel_counts'])
        value['pixel_fractions_percent'] = [100 * n / total for n in value['pixel_counts']]
    inputs[str(audit_path)] = sha(audit_path)
    by_id = {m['id']: m for m in models}
    comparisons = {}
    for seed in ['3407', '3408']:
        metrics = {k: by_id[k]['validation_seed' + seed]['miou_4class'] for k in ['U', 'H', 'R', 'HR']}
        comparisons[seed] = dict(high_noise_without_recycling_pp=100 * (metrics['H'] - metrics['U']),
                                 high_noise_with_recycling_pp=100 * (metrics['HR'] - metrics['R']),
                                 recycling_uniform_pp=100 * (metrics['R'] - metrics['U']),
                                 recycling_high_noise_pp=100 * (metrics['HR'] - metrics['H']),
                                 joint_vs_original_pp=100 * (metrics['HR'] - metrics['U']),
                                 interaction_pp=100 * (metrics['HR'] - metrics['H'] - metrics['R'] + metrics['U']))
    costs = {k: (m['train_seconds_sum'] + m['training_validation_seconds_sum']) / 3600 for k, m in by_id.items()}
    projected = dict(
        note='Linear train+validation estimates from recorded40-epoch logs; excludes setup, checkpoint/log I/O, new baselines and final evaluation. Not a scheduling promise.',
        six_40epoch_main_repeat_runs_hours=2 * sum(costs[k] for k in ['D', 'I', 'HR']),
        six_40epoch_remaining_factorial_repeat_runs_hours=2 * sum(costs[k] for k in ['U', 'H', 'R']),
        twelve_40epoch_repeat_runs_hours=2 * sum(costs[k] for k in ['D', 'I', 'U', 'H', 'R', 'HR']),
        twelve_200epoch_four_model_three_seed_runs_hours=5 * 3 * sum(costs[k] for k in ['D', 'I', 'U', 'HR']),
        eighteen_200epoch_six_model_three_seed_runs_hours=5 * 3 * sum(costs[k] for k in ['D', 'I', 'U', 'H', 'R', 'HR']))
    inventory = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), scope='Artifact-only inventory; no training/inference/test evaluation.',
                     completed_p1_trainings=7, p1_optimizer_updates=16800, distinct_training_seeds=[42],
                     independent_replicates_per_method=1, completed_test_models=3,
                     p2_formal_training_count=0, models=models, dataset=dataset,
                     factorial_effects_pp=comparisons, projected_costs=projected,
                     test_exposure='Existing27-image test results have been examined; no future tuning/selection on these scores. New independent confirmation requires a genuinely unseen holdout or external samples.',
                     evaluation_source_sha256=current, inputs_sha256=inputs)
    (OUT / 'experiment_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'experiment_inventory.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'projected_costs.json').write_text(json.dumps(projected, indent=2) + '\n')
    lines = ['# 第3章已有实验：核实清单', '', '2026-10-18。只读取已有产物，无训练、推理或新测试。', '',
             '| ID | 方法 | best轮次 | val mIoU3407% | val mIoU3408% | test mIoU% |',
             '|---|---|---:|---:|---:|---:|']
    for r in rows:
        secondary = '—' if r['val_miou_3408_percent'] == '' else f"{r['val_miou_3408_percent']:.2f}"
        test = '未评估' if r['test_miou_percent'] == '' else f"{r['test_miou_percent']:.2f}"
        lines.append(f"| {r['id']} | {r['name']} | {r['best_epoch']} | {r['val_miou_3407_percent']:.2f} | {secondary} | {test} |")
    lines += ['', '7次P1均为40轮/2400更新/训练seed42，每种方法只有1个独立训练重复。3408为同一个3407-selected best的采样复核；不是第2次训练。P2未启动。', '',
              '完整数值、指标来源、输入哈希及历次评估协议见 experiment_inventory.json；CSV可直接用于制表。projected_costs.json为日志线性估计，不是已运行的新实验。', '',
              '已有与待补内容的判断及执行顺序见 docs/CHAPTER3_EVIDENCE_AND_SUPPLEMENT_PLAN_20261018.md。', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    assert all(sha(p) == digest for p, digest in inputs.items())
    print('\n'.join(lines))
    print(json.dumps(projected, indent=2))


if __name__ == '__main__':
    main()

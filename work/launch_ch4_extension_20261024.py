"""Coordinator for the Chapter-4 extension batch (docs/CHAPTER4_FIGURES_TABLES_AND_TRAINING_PLAN_20261024.md, Section 7).

Order: lock -> PhysicalNet source-hash check -> freeze plan.json ->
  stage 1  E3R val25 process export          (work/export_e3r_process_20261024.py)
  stage 2  RepELA engineering verification   (work/repela_20261024/verify.py)
  stage 3  RepELA unified training, then test27 once (training form) and deploy-form val25/test27 equivalence
  stage 4  unified inference-cost benchmark  (work/benchmark_ch4_20261024.py)
  report   comparison_summary.json, final_analysis.md; PhysicalNet hashes re-checked; stop (no plotting).
Rerun the same command to resume: completed stages are skipped, training resumes from last.pt.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import json
import fcntl
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

WORK_DIR = Path(__file__).resolve().parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

import numpy as np
import torch
from PIL import Image

import glpflow_core as G
from conditional_diffusion.data import PALETTE
from repela_20261024 import core as R

B = R.B
PY = sys.executable
ROOT = G.ROOT
PLAN = R.BATCH / 'plan.json'
EXPORT_REPORT = ROOT / 'outputs/analysis/e3r_process_export_20261024/report.json'
VERIFY_REPORT = R.BATCH / 'engineering_verification_repela.json'
BENCH = ROOT / 'outputs/analysis/ch4_benchmark_20261024'
PHYSICALNET_SHA256 = {
    'repela_net.py': '9f864363825775c1537dd4a5f01620125f1d14fc10cecdfbe3563d6c79b96f3d',
    'rep_conv.py': 'dd6b6739c82bb6c2b0cda6dff92b5b65957d7610f1f3f1eacd7c6b270c217d34',
    'ela_block.py': '90bb164dd2d9895722ae86afe99ae58a7b767bc6ff5c1a2b66795d31bb7d7870',
    'decoder.py': 'e17ad4c639222514e39325751222c57d18940ba8f15a693ae1492a99c77c3a30',
}
FROZEN = [
    'work/repela_20261024/__init__.py',
    'work/repela_20261024/model.py',
    'work/repela_20261024/core.py',
    'work/repela_20261024/verify.py',
    'work/repela_20261024/physicalnet_models/__init__.py',
    *[f'work/repela_20261024/physicalnet_models/{f}' for f in PHYSICALNET_SHA256],
    'work/export_e3r_process_20261024.py',
    'work/benchmark_ch4_20261024.py',
    'work/launch_ch4_extension_20261024.py',
    'work/glpflow_core.py',
    'work/diagnose_glpflow_20261022.py',
    'src/conditional_diffusion/data.py',
    'src/conditional_diffusion/inference.py',
    'src/conditional_diffusion/metrics_impl.py',
    'src/conditional_diffusion/runtime.py',
    'work/baselines_20261022/models.py',
    'work/baselines_20261022/diffusion.py',
    'work/baselines_20261022/core.py',
    'data_manifests/source_dataset.json',
    'data_manifests/p0_fixed_crops.json',
    'outputs/launches/p1_baselines_20261022/plan.json',
    'data_manifests/splits/train.txt',
    'data_manifests/splits/val.txt',
    'data_manifests/splits/test.txt',
]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def status(**kw):
    p = R.BATCH / 'status.json'
    s = json.loads(p.read_text()) if p.exists() else {}
    s.update(kw, updated_bjt=now())
    G.atomic_json(p, s)


def check_physicalnet():
    for f, h in PHYSICALNET_SHA256.items():
        cur = G.digest(R.SOURCE_DIR / f)
        assert cur == h, f'PhysicalNet source changed: {f}'
        assert G.digest(R.COPY_DIR / f) == h, f'copy differs from source: {f}'


def existing_checkpoints():
    base = json.loads((ROOT / 'outputs/launches/p1_baselines_20261022/plan.json').read_text())['runs']
    ck = {m: Path(r) / 'checkpoints/best.pt' for m, r in base.items()}
    for arm, r in (('E0', 'p1_glpflow_20261021_e0'), ('E1', 'p1_glpflow_20261021_e1'), ('E3R', 'p1_glpflow_rollout_20261022_e3r')):
        ck[arm] = ROOT / f'outputs/P1/glpflow/seed_42/{r}/checkpoints/best.pt'
    return ck


def freeze():
    if PLAN.exists():
        return json.loads(PLAN.read_text())
    plan = dict(
        batch_id='p1_ch4_extension_20261024',
        created_bjt=now(),
        document='docs/CHAPTER4_FIGURES_TABLES_AND_TRAINING_PLAN_20261024.md (Section 7)',
        stages=['export_e3r_process', 'verify_repela', 'train_repela', 'test_repela', 'benchmark', 'report'],
        repela_config=R.config(),
        repela_run=str(R.RUN),
        repela_test_output=str(R.TEST_OUT),
        frozen_sha256={p: G.digest(ROOT / p) for p in FROZEN},
        existing_checkpoints_sha256={k: G.digest(v) for k, v in existing_checkpoints().items()},
        physicalnet_sha256=PHYSICALNET_SHA256,
        scope='seed42, single RepELA-Net-Small run from scratch under the unified protocol; val25 selection; test27 once; '
              'no retraining of existing models; PhysicalNet read-only',
    )
    G.atomic_json(PLAN, plan)
    return plan


def guard(plan):
    for p, h in plan['frozen_sha256'].items():
        assert G.digest(ROOT / p) == h, f'Frozen file modified during execution: {p}'


def guard_stage(plan):
    guard(plan)
    check_physicalnet()
    checkpoints = existing_checkpoints()
    for k, h in plan['existing_checkpoints_sha256'].items():
        assert G.digest(checkpoints[k]) == h, f'checkpoint changed: {k}'


def run_sub(args, log_name, stage):
    status(stage=stage)
    with open(R.BATCH / log_name, 'a') as log:
        rc = subprocess.call([PY, '-B', '-W', 'ignore', *args], stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        status(stage=f'failed_{stage}', returncode=rc)
        sys.exit(rc)


# ----------------------------------------------------------------------------- stage 3 workers

def worker_train():
    plan = json.loads(PLAN.read_text())
    guard(plan)
    c = plan['repela_config']
    G.configure(c['seed'])
    res = B.train(c, plan['repela_run'], G.digest(PLAN), guard_fn=lambda: guard(plan))
    print(res, flush=True)


def _image_miou(m):
    i = np.diag(m).astype(float)
    u = m.sum(0) + m.sum(1) - np.diag(m)
    v = u > 0
    return float((i[v] / u[v]).mean())


def worker_test():
    plan = json.loads(PLAN.read_text())
    guard(plan)
    out = R.TEST_OUT
    assert shutil.disk_usage(ROOT).free > 8 * 1024**3
    out.mkdir(parents=True, exist_ok=True)
    run = Path(plan['repela_run'])
    ck = run / 'checkpoints/best.pt'
    val = json.loads((run / 'best_val_metrics.json').read_text())
    state = torch.load(ck, map_location='cpu', weights_only=False)
    c = state['config']
    assert c == plan['repela_config'] and val['epoch'] == state['epoch']
    tp = out / 'test_plan.json'
    if not tp.exists():
        G.atomic_json(tp, dict(created_bjt=now(), split='test', historical_test_exposure=True,
                               selected_by='validation-best EMA checkpoint; no test reselection',
                               checkpoint=dict(path=str(ck), sha256=G.digest(ck)), plan_sha256=G.digest(PLAN)))
    man = G.read_manifest(c['manifest'])
    root = Path(man['dataset_root'])
    results = {}

    # (1) training form: test27 once, masks saved and re-verified
    G.configure(c['seed'])
    model = B.build_model(c).cuda()
    model.load_state_dict(state['model'], strict=True)
    mdir, cdir = out / 'train_form/masks', out / 'train_form/color'
    mdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)

    def save(ident, pred):
        Image.fromarray(pred).save(mdir / f'{ident}.png', compress_level=3)
        Image.fromarray(PALETTE[pred]).save(cdir / f'{ident}.png', compress_level=3)

    tm, mats, _ = B.evaluate(model, c, split='test', on_image=save)
    assert tm['image_count'] == 27
    rec = []
    for ident in mats['ids']:
        with Image.open(root / man['mask_pattern'].format(id=ident)) as a, Image.open(mdir / f'{ident}.png') as b:
            rec.append(G.confusion(np.array(a), np.array(b)))
    tm['saved_masks_match_confusion'] = bool(np.array_equal(np.array(rec), mats['per_image']))
    assert tm['saved_masks_match_confusion']
    np.savez_compressed(out / 'train_form_confusion.npz', **mats)
    G.atomic_json(out / 'per_image_miou.json', [dict(method='repela_small', form='train', image_id=str(i), miou_present=_image_miou(m))
                                                 for i, m in zip(mats['ids'], mats['per_image'])])
    results['train_form'] = dict(best_epoch=val['epoch'], val=val['metrics'], test=tm)

    # (2) deploy form (same weights, RepConv branches fused): val25 and test27 equivalence
    model.switch_to_deploy()
    agree = []

    def compare(ident, pred):
        with Image.open(mdir / f'{ident}.png') as b:
            agree.append(float((np.array(b) == pred).mean()))

    dv, dvm, _ = B.evaluate(model, c, split='val')
    dt, dtm, _ = B.evaluate(model, c, split='test', on_image=compare)
    ref_val = np.load(run / 'best_val_confusion.npz')
    np.savez_compressed(out / 'deploy_form_val_confusion.npz', **dvm)
    np.savez_compressed(out / 'deploy_form_test_confusion.npz', **dtm)
    results['deploy_form'] = dict(
        params=sum(p.numel() for p in model.parameters()), val=dv, test=dt,
        val_confusion_identical_to_train_form=bool(np.array_equal(dvm['per_image'], ref_val['per_image'])),
        test_confusion_identical_to_train_form=bool(np.array_equal(dtm['per_image'], mats['per_image'])),
        val_miou_diff_pp=100 * (dv['miou_4class'] - val['metrics']['miou_4class']),
        test_miou_diff_pp=100 * (dt['miou_4class'] - tm['miou_4class']),
        test_pixel_agreement_min=min(agree), test_pixel_agreement_mean=float(np.mean(agree)))
    G.atomic_json(out / 'test_metrics.json', results)
    print('[TEST repela]', json.dumps({k: results['deploy_form'][k] for k in results['deploy_form'] if k not in ('val', 'test')}), flush=True)


# ----------------------------------------------------------------------------- report

def report(plan):
    run = Path(plan['repela_run'])
    tr = json.loads((R.TEST_OUT / 'test_metrics.json').read_text())
    bench = json.loads((BENCH / 'benchmark.json').read_text())
    ex = json.loads(EXPORT_REPORT.read_text())
    ver = json.loads(VERIFY_REPORT.read_text())
    meta = json.loads((run / 'run_metadata.json').read_text())
    summary = dict(batch_id=plan['batch_id'], completed_bjt=now(), export=dict(passed=ex['all_checks_passed'], val_miou=ex['val_miou_4class'], size_mib=ex['size_mib']),
                   repela=dict(parameters=meta['parameters'], verification_passed=ver['passed'], **tr), benchmark=bench)
    G.atomic_json(R.BATCH / 'comparison_summary.json', summary)
    t, v, dp = tr['train_form']['test'], tr['train_form']['val'], tr['deploy_form']
    pct = lambda x: f'{100 * x:.2f}'
    lines = [
        '# 第四章扩展批次汇总（2026-10-24）', '',
        '## 1. E3R 过程导出（val25）', '',
        f"- 全部校验通过：{ex['all_checks_passed']}；val mIoU {pct(ex['val_miou_4class'])}%；大小 {ex['size_mib']} MiB。", '',
        '## 2. RepELA-Net-Small 统一协议结果（第五章使用）', '',
        f"- 参数 {meta['parameters']:,}（部署形态 {dp['params']:,}）；val best 第 {tr['train_form']['best_epoch']} 轮；工程验证通过：{ver['passed']}。",
        '', '| 形态 | val mIoU | val 单层/少层/多层 IoU | val 前景边界 F1 | test mIoU | test 单层/少层/多层 IoU | test 前景边界 F1 |',
        '|---|---:|---|---:|---:|---|---:|',
    ]
    for name, vv, tt in (('训练形态', v, t), ('部署形态', dp['val'], dp['test'])):
        lines.append(f"| {name} | {pct(vv['miou_4class'])} | {pct(vv['iou_monolayer'])} / {pct(vv['iou_fewlayer'])} / {pct(vv['iou_multilayer'])} | "
                     f"{pct(vv['boundary_f1_foreground'])} | {pct(tt['miou_4class'])} | {pct(tt['iou_monolayer'])} / {pct(tt['iou_fewlayer'])} / "
                     f"{pct(tt['iou_multilayer'])} | {pct(tt['boundary_f1_foreground'])} |")
    lines += ['', f"- 部署等价性：val 混淆矩阵一致 {dp['val_confusion_identical_to_train_form']}（差 {dp['val_miou_diff_pp']:.4f}pp），"
                  f"test 一致 {dp['test_confusion_identical_to_train_form']}（差 {dp['test_miou_diff_pp']:.4f}pp），test 逐图像素一致率最小 {dp['test_pixel_agreement_min']:.6f}。",
              '', '## 3. 统一推理成本（RTX 3090，FP32）', '',
              '| 对象 | 步数 | 参数 (M) | GMACs/窗 | GMACs/图 | 窗口延迟 中位/P95 (ms) | 吞吐 b8 (窗/s) | 整图 预测/网络 (s) | 推理峰值显存 alloc/reserved (MiB) |',
              '|---|---:|---:|---:|---:|---|---:|---|---|']
    for r in bench['objects']:
        img = f"{r['image_predict_s_median']:.2f} / {r['image_network_s_median']:.2f}" if 'image_predict_s_median' in r else '-'
        mem = f"{r['infer_peak_allocated_mib']:.0f} / {r['infer_peak_reserved_mib']:.0f}" if 'infer_peak_allocated_mib' in r else '-'
        lines.append(f"| {r['name']} | {r['steps']} | {r['params'] / 1e6:.2f} | {r['window_gmacs']:.2f} | {r['image_gmacs']:.1f} | "
                     f"{r['window_latency_ms_median']:.1f} / {r['window_latency_ms_p95']:.1f} | {r['window_throughput_per_s_b8']:.1f} | {img} | {mem} |")
    lines += ['', f"- 原评估代码与原生算子预测一致率低于 99.99% 的对象：{bench['agreement_below_99_99'] or '无'}。",
              '- 口径见 `outputs/analysis/ch4_benchmark_20261024/README.md`。test27 历史已暴露，非盲测；单训练种子 42。']
    (R.BATCH / 'final_analysis.md').write_text('\n'.join(lines) + '\n')


def coordinator_locked():
    status(stage='preflight', pid=os.getpid())
    assert shutil.disk_usage(ROOT).free >= 15 * 1024**3, 'need >=15 GiB free before starting'
    check_physicalnet()
    plan = freeze()
    guard_stage(plan)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.split()
    assert not apps, f'GPU is busy: {apps}'
    status(stage='starting', pid=os.getpid(), plan_sha256=G.digest(PLAN))
    os.environ['CH4_COORDINATOR_HOLDS_LOCK'] = '1'  # children run under this process's experiments lock

    if not (EXPORT_REPORT.exists() and json.loads(EXPORT_REPORT.read_text()).get('all_checks_passed')):
        run_sub([str(WORK_DIR / 'export_e3r_process_20261024.py')], 'export_e3r_process.log', 'export_e3r_process')
    guard_stage(plan)

    if not (VERIFY_REPORT.exists() and json.loads(VERIFY_REPORT.read_text()).get('passed')):
        run_sub([str(WORK_DIR / 'repela_20261024/verify.py')], 'verify_repela.log', 'verify_repela')
    guard_stage(plan)

    st = R.RUN / 'status.json'
    if not (st.exists() and json.loads(st.read_text()).get('status') == 'completed'):
        run_sub([__file__, '--worker-train'], 'train_repela_small.log', 'train_repela')
    guard_stage(plan)

    if not (R.TEST_OUT / 'test_metrics.json').exists():
        run_sub([__file__, '--worker-test'], 'test_repela_small.log', 'test_repela')
    guard_stage(plan)

    if not (BENCH / 'benchmark.json').exists():
        run_sub([str(WORK_DIR / 'benchmark_ch4_20261024.py')], 'benchmark.log', 'benchmark')
    guard_stage(plan)

    status(stage='report')
    report(plan)
    check_physicalnet()
    status(stage='completed', finished_bjt=now())
    print('CH4 EXTENSION BATCH COMPLETED', flush=True)


def coordinator():
    R.BATCH.mkdir(parents=True, exist_ok=True)
    lock = open(ROOT / 'outputs/.experiments.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        coordinator_locked()
    except Exception as exc:
        current = json.loads((R.BATCH / 'status.json').read_text()).get('stage', 'preflight')
        status(stage=f'failed_{current}', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--worker-train':
        worker_train()
    elif len(sys.argv) == 2 and sys.argv[1] == '--worker-test':
        worker_test()
    else:
        coordinator()

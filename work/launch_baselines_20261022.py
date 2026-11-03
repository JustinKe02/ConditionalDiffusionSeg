"""Coordinator for the Chapter-3 comparison baselines batch (2026-10-22).

Execution order (per docs/TRAINING_PLAN_BASELINES_20261022.md Section 7):
  1. Verify engineering_verification.json passed and freeze plan.json.
  2. Acquire exclusive lock outputs/.experiments.lock.
  3. Serial training: unet -> unetpp -> deeplabv3p -> pspnet -> segformer_b1 -> hrnet_w48 -> mask2former -> ddp -> ccdm.
  4. Secondary val25 evaluation with sampling seed 3408 for ddp and ccdm best checkpoints.
  5. Test27 inference once per method using val-best EMA checkpoints, saving masks, color maps, and confusion matrices,
     and verifying saved masks match per-image confusion matrices pixel-for-pixel.
  6. Generate comparison_summary.json and final_analysis.md (including read-only GLP-Flow E0, E1, E3R val & test results).

Rerun the same command to resume; completed methods are skipped based on status.json, incomplete methods resume from last.pt.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import json
import time
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
from baselines_20261022 import core as C

PY = sys.executable
PLAN = C.BATCH / 'plan.json'
PLAN_V1 = C.BATCH / 'plan_v1_20261022.json'  # superseded plan, kept for the record (revision 2026-10-23)
FROZEN = [
    'work/baselines_20261022/__init__.py',
    'work/baselines_20261022/models.py',
    'work/baselines_20261022/diffusion.py',
    'work/baselines_20261022/core.py',
    'work/baselines_20261022/verify.py',
    'work/launch_baselines_20261022.py',
    'outputs/launches/p1_baselines_20261022/sources.json',
    'outputs/launches/p1_baselines_20261022/engineering_verification.json',
    'data_manifests/source_dataset.json',
    'data_manifests/splits/train.txt',
    'data_manifests/splits/val.txt',
    'data_manifests/splits/test.txt',
]


def now() -> str:
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def status(**kw):
    st_path = C.BATCH / 'status.json'
    s = json.loads(st_path.read_text()) if st_path.exists() else {}
    s.update(kw, updated_bjt=now())
    G.atomic_json(st_path, s)


def freeze():
    if PLAN.exists():
        return json.loads(PLAN.read_text())
    v = json.loads((C.BATCH / 'engineering_verification.json').read_text())
    assert v.get('passed'), 'engineering verification must pass before freezing plan.json'
    grad_accum = v.get('grad_accum_selected', {})
    for m, acc in grad_accum.items():
        if m in C.METHODS:
            C.METHODS[m]['grad_accum_steps'] = int(acc)

    configs = {m: C.config(m) for m in C.ORDER}
    runs = {m: str(C.RUN_ROOT / f'p1_baselines_20261022_{m}') for m in C.ORDER}
    plan = dict(
        batch_id='p1_baselines_20261022',
        created_bjt=now(),
        order=C.ORDER,
        configs=configs,
        runs=runs,
        test_output=str(C.TEST_OUT),
        frozen_sha256={p: G.digest(C.ROOT / p) for p in FROZEN},
        scope='seed42 single training run per method from scratch (no pretrained weights, no DINO); original train120/val25 incl. m109/m31; test27 evaluated once at the end',
        budget='100 epochs x 240 crops = 24000 crops, effective batch 8, 3000 updates per method; validation every 10 epochs (PowerEMA sigma_rel=0.10)',
        selection='best by global 4-class val25 mIoU (EMA weights, sampling seed 3407); 3408 same-best check for ddp and ccdm; single test27 evaluation using val-best',
    )
    if PLAN_V1.exists():
        plan['supersedes'] = dict(
            file=str(PLAN_V1.relative_to(C.ROOT)),
            sha256=G.digest(PLAN_V1),
            reason='2026-10-23 revision: Mask2Former restored to the original objective (no extra semantic loss); '
                   'DDP/CCDM grad_accum_steps=2 as measured by verification check 3 (v1 recorded 1 because check 3 '
                   'was skipped on a verification resume). Methods completed under v1 (unet, unetpp, deeplabv3p, '
                   'pspnet, segformer_b1) use code unaffected by these changes and are not retrained.',
        )
    G.atomic_json(PLAN, plan)
    return plan


def guard(plan: dict):
    for p, h in plan['frozen_sha256'].items():
        cur = G.digest(C.ROOT / p)
        assert cur == h, f'Frozen file modified during execution: {p} (expected {h}, got {cur})'


def worker(method: str):
    plan = json.loads(PLAN.read_text())
    guard(plan)
    c = plan['configs'][method]
    G.configure(c['seed'])
    res = C.train(c, plan['runs'][method], G.digest(PLAN), guard_fn=lambda: guard(plan))
    print(res, flush=True)


def secondary(method: str):
    plan = json.loads(PLAN.read_text())
    guard(plan)
    c = plan['configs'][method]
    run = Path(plan['runs'][method])
    G.configure(c['seed'])
    state = torch.load(run / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model = C.build_model(c).cuda()
    model.load_state_dict(state['model'], strict=True)
    metrics, mats, _ = C.evaluate(model, c, split='val', sampling_seed=3408)
    G.atomic_json(C.BATCH / f'secondary_{method}_3408.json', dict(epoch=state['epoch'], metrics=metrics))
    np.savez_compressed(C.BATCH / f'secondary_{method}_3408.npz', **mats)
    del model
    torch.cuda.empty_cache()


def _image_miou(m: np.ndarray) -> float:
    i = np.diag(m).astype(float)
    u = m.sum(0) + m.sum(1) - np.diag(m)
    v = u > 0
    return float((i[v] / u[v]).mean())


def run_test_eval(plan: dict):
    guard(plan)
    assert shutil.disk_usage(C.ROOT).free > 8 * (1024**3), 'Insufficient free disk space (<8 GiB)'
    C.TEST_OUT.mkdir(parents=True, exist_ok=True)
    ckpt = {m: Path(plan['runs'][m]) / 'checkpoints/best.pt' for m in C.ORDER}
    test_plan_path = C.TEST_OUT / 'test_plan.json'
    if not test_plan_path.exists():
        test_plan = dict(
            created_bjt=now(),
            authorization='Authorized in docs/TRAINING_PLAN_BASELINES_20261022.md Section 7.6',
            split='test',
            historical_test_exposure=True,
            selected_by='validation-best EMA checkpoint; no test reselection',
            sampling_seed=3407,
            plan_sha256=G.digest(PLAN),
            test_split_sha256=G.digest(C.ROOT / 'data_manifests/splits/test.txt'),
            checkpoints={m: dict(path=str(p), sha256=G.digest(p)) for m, p in ckpt.items()},
        )
        G.atomic_json(test_plan_path, test_plan)

    metrics_path = C.TEST_OUT / 'test_metrics.json'
    results = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    per_img_path = C.TEST_OUT / 'per_image_miou.json'
    rows = json.loads(per_img_path.read_text()) if per_img_path.exists() else []

    man = G.read_manifest(plan['configs']['unet']['manifest'])
    ds_root = Path(man['dataset_root'])

    for m in C.ORDER:
        if m in results and (C.TEST_OUT / f'{m}_confusion.npz').exists():
            continue
        status(stage=f'testing_{m}')
        path = ckpt[m]
        state = torch.load(path, map_location='cpu', weights_only=False)
        c = state['config']
        assert c['method'] == m and c['sampling_seed'] == 3407
        val = json.loads((Path(plan['runs'][m]) / 'best_val_metrics.json').read_text())
        assert val['epoch'] == state['epoch']

        G.configure(c['seed'])
        model = C.build_model(c).cuda()
        model.load_state_dict(state['model'], strict=True)
        del state

        mask_dir = C.TEST_OUT / m / 'masks'
        color_dir = C.TEST_OUT / m / 'color'
        mask_dir.mkdir(parents=True, exist_ok=True)
        color_dir.mkdir(parents=True, exist_ok=True)

        def save_img(ident, pred, mask_dir=mask_dir, color_dir=color_dir):
            Image.fromarray(pred).save(mask_dir / f'{ident}.png', compress_level=3)
            Image.fromarray(PALETTE[pred]).save(color_dir / f'{ident}.png', compress_level=3)

        t_metrics, mats, _ = C.evaluate(model, c, split='test', on_image=save_img)
        assert t_metrics['image_count'] == 27

        # Verify saved masks match confusion matrix image-by-image (Section 7.6)
        recomputed = []
        for ident in mats['ids']:
            with Image.open(ds_root / man['mask_pattern'].format(id=ident)) as im_gt:
                gt_arr = np.array(im_gt)
            with Image.open(mask_dir / f'{ident}.png') as im_pr:
                pr_arr = np.array(im_pr)
            recomputed.append(G.confusion(gt_arr, pr_arr))
        recomputed_arr = np.array(recomputed)
        masks_match = bool(np.array_equal(recomputed_arr, mats['per_image']))
        assert masks_match, f'Saved masks do not match confusion matrix for {m}'
        t_metrics['saved_masks_match_confusion'] = masks_match

        np.savez_compressed(C.TEST_OUT / f'{m}_confusion.npz', **mats)
        rows = [r for r in rows if r.get('method') != m]
        for ident, mat in zip(mats['ids'], mats['per_image']):
            rows.append(dict(method=m, image_id=str(ident), miou_present=_image_miou(mat)))
        G.atomic_json(per_img_path, rows)

        results[m] = dict(best_epoch=val['epoch'], test=t_metrics, val=val['metrics'])
        G.atomic_json(metrics_path, results)
        print(f"[TEST {m}] test mIoU={100*t_metrics['miou_4class']:.2f}% val={100*val['metrics']['miou_4class']:.2f}% sec={t_metrics['seconds']:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    return results


def _load_glpflow_reference():
    """Read existing GLP-Flow E0, E1, E3R val and test results without re-running (Section 7.7)."""
    glp_runs = {
        'GLP-Flow E0': C.ROOT / 'outputs/P1/glpflow/seed_42/p1_glpflow_20261021_e0',
        'GLP-Flow E1': C.ROOT / 'outputs/P1/glpflow/seed_42/p1_glpflow_20261021_e1',
        'GLP-Flow E3R': C.ROOT / 'outputs/P1/glpflow/seed_42/p1_glpflow_rollout_20261022_e3r',
    }
    arm_key = {'GLP-Flow E0': 'E0', 'GLP-Flow E1': 'E1', 'GLP-Flow E3R': 'E3R'}
    glp_test = json.loads((C.ROOT / 'outputs/inference/glpflow_test_20261022/test_metrics.json').read_text())
    glp_mech = json.loads((C.ROOT / 'outputs/analysis/glpflow_e3r_mechanism_20261022/report.json').read_text())['runs']

    out = {}
    for disp, rdir in glp_runs.items():
        ak = arm_key[disp]
        meta = json.loads((rdir / 'run_metadata.json').read_text())
        val_best = json.loads((rdir / 'best_val_metrics.json').read_text())
        vm = val_best['metrics']
        tm = glp_test[ak]['test']
        mech_val = glp_mech['E3R_predict_k4' if ak == 'E3R' else ak]
        mech_test = glp_mech[f'test_{ak}']
        out[disp] = dict(
            display=disp,
            parameters=meta['parameters'],
            best_epoch=val_best['epoch'],
            val=dict(
                miou_4class=vm['miou_4class'],
                iou_monolayer=vm['iou_monolayer'],
                iou_fewlayer=vm['iou_fewlayer'],
                iou_multilayer=vm['iou_multilayer'],
                thin_mean_iou=vm['thin_mean_iou'],
                boundary_f1_foreground=vm['boundary_f1_foreground'],
                acc_0_2px_all=mech_val['distance_accuracy']['0-2']['all'],
                acc_0_2px_thin=mech_val['distance_accuracy']['0-2']['thin'],
            ),
            test=dict(
                miou_4class=tm['miou_4class'],
                iou_monolayer=tm['iou_monolayer'],
                iou_fewlayer=tm['iou_fewlayer'],
                iou_multilayer=tm['iou_multilayer'],
                thin_mean_iou=tm['thin_mean_iou'],
                boundary_f1_foreground=tm['boundary_f1_foreground'],
                acc_0_2px_all=mech_test['distance_accuracy']['0-2']['all'],
                acc_0_2px_thin=mech_test['distance_accuracy']['0-2']['thin'],
            ),
            nfe_per_window=4 if ak == 'E3R' else 1,
            nfe_per_image=int(tm['nfe_total'] // tm['image_count']),
            predict_seconds_per_image=mech_val['seconds_per_image'],
            peak_reserved_mib=mech_val['peak_reserved_mib'],
        )
    return out


def generate_report(plan: dict, test_results: dict):
    res = {}
    for m in C.ORDER:
        run = Path(plan['runs'][m])
        best = json.loads((run / 'best_val_metrics.json').read_text())
        meta = json.loads((run / 'run_metadata.json').read_text())
        z = np.load(run / 'best_val_confusion.npz')
        keep = [i for i, s in enumerate(z['ids']) if str(s) != 'm109']
        no109 = G.classification_metrics(z['per_image'][keep].sum(0))
        vm = best['metrics']
        tm = test_results[m]['test']
        entry = dict(
            method=m,
            display=plan['configs'][m]['display'],
            parameters=meta['parameters'],
            grad_accum_steps=meta['grad_accum_steps'],
            bn_micro_batch_size=meta['bn_micro_batch_size'],
            best_epoch=best['epoch'],
            val=vm,
            test=tm,
            descriptive_val_without_m109=dict(miou=no109['miou_4class'], thin_mean_iou=no109['thin_mean_iou']),
            nfe_per_window=plan['configs'][m]['nfe_per_window'],
            nfe_per_image=int(tm['nfe_per_image']),
            predict_seconds_per_image=vm['predict_seconds_per_image'],
            test_predict_seconds_per_image=tm['predict_seconds_per_image'],
            train_peak_reserved_mib=vm.get('train_peak_reserved_mib'),
            # Inference peak from the test27 pass (fresh model after empty_cache); the validation value inside
            # training only reflects the allocator cache left by training.
            infer_peak_reserved_mib=tm['peak_reserved_mib'],
        )
        sec_f = C.BATCH / f'secondary_{m}_3408.json'
        if sec_f.exists():
            entry['val_seed3408'] = json.loads(sec_f.read_text())['metrics']
        res[m] = entry

    glp_ref = _load_glpflow_reference()
    summary = dict(
        batch_id=plan['batch_id'],
        completed_bjt=now(),
        scope=plan['scope'],
        budget=plan['budget'],
        baselines=res,
        glpflow_reference=glp_ref,
    )
    G.atomic_json(C.BATCH / 'comparison_summary.json', summary)

    lines = [
        '# 第3章对比方法训练与评估汇总报告（2026-10-22）',
        '',
        '## 1. 协议与说明',
        '- 全部 9 个对比方法均在同一框架、同一数据划分（train120 / val25 / test27，保留 m109/m31）下**从头训练**（无预训练权重，不含 DINO），训练种子 42。',
        '- 预算：100 轮 × 240 裁块 = 24000 裁块，有效 batch 8，3000 次参数更新；AdamW lr=2.8e-4，warmup 5 轮后余弦衰减至 1e-6，PowerEMA（σ_rel=0.10）。',
        '- 选模：每 10 轮在 val25 全分辨率图（512 窗口 / 384 步长）上评估 EMA 权重，按全局四类 val mIoU 选 best；扩散方法以采样种子 3407 选模，并用 3408 对同一 best 复核一次。',
        '- 测试：全部训练完成后，各方法用各自 val best EMA 权重在 test27 上评估一次（历史 test 已暴露，非盲测，未按 test 重选）。',
        '- 纯预测耗时口径与 §62 一致：仅计 `predict`（含 CPU 滑窗概率融合，不含读图与指标计算）。',
        '- 峰值显存：训练列为训练轮内峰值；推理列取 test27 评估阶段（单独加载 best 权重并清空缓存后）的峰值。',
        '',
        '## 2. 汇总表（表3-5 数据来源）',
        '',
        '| 方法 | 参数量 (M) | BN batch | val best 轮 | val mIoU (%) | val 单层/少层/多层 IoU (%) | val 前景边界 F1 (%) | test mIoU (%) | test 单层/少层/多层 IoU (%) | test 前景边界 F1 (%) | 边界 0–2px 准确率 val/test (%) | NFE (每窗/每图) | 每图纯预测耗时 (s) | 峰值显存 训练/推理 (MiB) |',
        '|---|---:|---:|---:|---:|---|---:|---:|---|---:|---|---|---:|---|',
    ]

    for m in C.ORDER:
        r = res[m]
        v = r['val']
        t = r['test']
        p_m = r['parameters'] / 1e6
        v_ious = f"{100*v['iou_monolayer']:.2f} / {100*v['iou_fewlayer']:.2f} / {100*v['iou_multilayer']:.2f}"
        t_ious = f"{100*t['iou_monolayer']:.2f} / {100*t['iou_fewlayer']:.2f} / {100*t['iou_multilayer']:.2f}"
        acc_02 = f"{100*v['acc_0_2px_all']:.2f} / {100*t['acc_0_2px_all']:.2f}"
        nfe_str = f"{r['nfe_per_window']} / {r['nfe_per_image']}"
        mem_str = f"{r['train_peak_reserved_mib']:.0f} / {r['infer_peak_reserved_mib']:.0f}" if r['train_peak_reserved_mib'] else f"- / {r['infer_peak_reserved_mib']:.0f}"
        lines.append(
            f"| {r['display']} | {p_m:.2f} | {r['bn_micro_batch_size']} | {r['best_epoch']} | "
            f"{100*v['miou_4class']:.2f} | {v_ious} | {100*v['boundary_f1_foreground']:.2f} | "
            f"{100*t['miou_4class']:.2f} | {t_ious} | {100*t['boundary_f1_foreground']:.2f} | "
            f"{acc_02} | {nfe_str} | {r['predict_seconds_per_image']:.2f} | {mem_str} |"
        )

    for disp in ('GLP-Flow E0', 'GLP-Flow E1', 'GLP-Flow E3R'):
        g = glp_ref[disp]
        v = g['val']
        t = g['test']
        p_m = g['parameters'] / 1e6
        v_ious = f"{100*v['iou_monolayer']:.2f} / {100*v['iou_fewlayer']:.2f} / {100*v['iou_multilayer']:.2f}"
        t_ious = f"{100*t['iou_monolayer']:.2f} / {100*t['iou_fewlayer']:.2f} / {100*t['iou_multilayer']:.2f}"
        acc_02 = f"{100*v['acc_0_2px_all']:.2f} / {100*t['acc_0_2px_all']:.2f}"
        nfe_str = f"{g['nfe_per_window']} / {g['nfe_per_image']}"
        lines.append(
            f"| **{disp}** | {p_m:.2f} | 8 | {g['best_epoch']} | "
            f"{100*v['miou_4class']:.2f} | {v_ious} | {100*v['boundary_f1_foreground']:.2f} | "
            f"{100*t['miou_4class']:.2f} | {t_ious} | {100*t['boundary_f1_foreground']:.2f} | "
            f"{acc_02} | {nfe_str} | {g['predict_seconds_per_image']:.2f} | - / {g['peak_reserved_mib']:.0f} |"
        )

    lines += [
        '',
        '## 3. 扩散方法采样种子 3408 复核（同一 val-best 权重）',
        '',
        '| 方法 | best 轮 | 3407 val mIoU (%) | 3408 val mIoU (%) | 3407 前景边界 F1 (%) | 3408 前景边界 F1 (%) |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for m in ('ddp', 'ccdm'):
        r = res[m]
        v3407 = r['val']
        v3408 = r.get('val_seed3408', v3407)
        lines.append(
            f"| {r['display']} | {r['best_epoch']} | {100*v3407['miou_4class']:.2f} | {100*v3408['miou_4class']:.2f} | "
            f"{100*v3407['boundary_f1_foreground']:.2f} | {100*v3408['boundary_f1_foreground']:.2f} |"
        )

    (C.BATCH / 'final_analysis.md').write_text('\n'.join(lines) + '\n')
    return summary


def coordinator(stop_after_method: str = None):
    C.BATCH.mkdir(parents=True, exist_ok=True)
    lock_file = open(C.ROOT / 'outputs/.experiments.lock', 'w')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = freeze()
    guard(plan)
    st_init = json.loads((C.BATCH / 'status.json').read_text()) if (C.BATCH / 'status.json').exists() else {}
    status(stage='starting', pid=os.getpid(), plan_sha256=G.digest(PLAN), completed_methods=st_init.get('completed_methods', {}))

    for m in C.ORDER:
        run = Path(plan['runs'][m])
        st_path = run / 'status.json'
        if st_path.exists() and json.loads(st_path.read_text()).get('status') == 'completed':
            best = json.loads((run / 'best_val_metrics.json').read_text())
            cur_st = json.loads((C.BATCH / 'status.json').read_text())
            cm = cur_st.get('completed_methods', {})
            if m not in cm:
                cm[m] = dict(
                    best_epoch=best['epoch'],
                    val_miou=best['metrics']['miou_4class'],
                    val_mono_iou=best['metrics']['iou_monolayer'],
                    val_few_iou=best['metrics']['iou_fewlayer'],
                    val_multi_iou=best['metrics']['iou_multilayer'],
                    val_thin_mean_iou=best['metrics']['thin_mean_iou'],
                    val_boundary_f1=best['metrics']['boundary_f1_foreground'],
                )
                status(completed_methods=cm)
            if stop_after_method == m:
                print(f"[SKIPPED/COMPLETED {m}] {json.dumps(cm[m], ensure_ascii=False)}", flush=True)
                return
            continue

        print(f"[START {m}] {now()}", flush=True)
        status(stage=f'training_{m}', current_method=m, started_method_bjt=now())
        with open(C.BATCH / f'train_{m}.log', 'a') as log:
            rc = subprocess.call([PY, '-B', '-W', 'ignore', __file__, '--worker', m], stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            status(stage=f'failed_{m}', returncode=rc)
            sys.exit(rc)

        best = json.loads((run / 'best_val_metrics.json').read_text())
        cur_st = json.loads((C.BATCH / 'status.json').read_text())
        cm = cur_st.get('completed_methods', {})
        cm[m] = dict(
            finished_bjt=now(),
            best_epoch=best['epoch'],
            val_miou=best['metrics']['miou_4class'],
            val_mono_iou=best['metrics']['iou_monolayer'],
            val_few_iou=best['metrics']['iou_fewlayer'],
            val_multi_iou=best['metrics']['iou_multilayer'],
            val_thin_mean_iou=best['metrics']['thin_mean_iou'],
            val_boundary_f1=best['metrics']['boundary_f1_foreground'],
            val_acc_0_2px_all=best['metrics']['acc_0_2px_all'],
            val_acc_0_2px_thin=best['metrics']['acc_0_2px_thin'],
            predict_seconds_per_image=best['metrics']['predict_seconds_per_image'],
            train_peak_reserved_mib=best['metrics'].get('train_peak_reserved_mib'),
            infer_peak_reserved_mib=best['metrics']['peak_reserved_mib'],
        )
        status(stage=f'completed_{m}', completed_methods=cm)
        print(f"[COMPLETED {m}] {json.dumps(cm[m], ensure_ascii=False)}", flush=True)
        if stop_after_method == m:
            guard(plan)
            return

    for m in ('ddp', 'ccdm'):
        if (C.BATCH / f'secondary_{m}_3408.json').exists():
            continue
        status(stage=f'secondary_{m}_3408')
        with open(C.BATCH / f'secondary_{m}.log', 'a') as log:
            rc = subprocess.call([PY, '-B', '-W', 'ignore', __file__, '--secondary', m], stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            status(stage=f'failed_secondary_{m}', returncode=rc)
            sys.exit(rc)

    status(stage='testing_test27')
    test_results = run_test_eval(plan)
    summary = generate_report(plan, test_results)
    guard(plan)
    status(stage='completed', finished_bjt=now())
    print('BATCH COMPLETED SUCCESSFULLY', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == '--secondary':
        secondary(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == '--stop-after-method':
        coordinator(stop_after_method=sys.argv[2])
    else:
        coordinator()

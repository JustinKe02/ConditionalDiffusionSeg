"""Authorized N2'/N3': screen at40, conditionally continue to100, then stop."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from sampler_aligned_core import (
    ROOT, BATCH, OLD_BATCH, TIMES, EARLIER, atomic_json, audit_metrics,
    digest, guard, progress, read, secondary, source_hashes, train,
)

MODES = ("N2", "N3")
WORKFLOW = ("sampler_aligned_core.py", "launch_sampler_aligned_20261021.py",
            "verify_sampler_aligned_20261021.py")


def configs():
    old = read(OLD_BATCH / "plan.json")
    result = {}
    for mode in MODES:
        c = dict(old["configs"][mode])
        c.update(experiment_variant="sampler_aligned_" + mode.lower(),
                 display_mode=mode + "_prime",
                 timestep_sampling="uniform_deployment_grid",
                 training_timesteps=list(TIMES), training_earlier_timesteps=list(EARLIER),
                 objective="0.5CE+0.5foregroundDice+1.0deployment_interval_KL",
                 terminal_readout_objective="t1: CE/Dice only; zero KL",
                 kl_reduction="per-image pixel mean; zero t1; mean over complete batch",
                 screen_epoch=40, epochs=100,
                 initialization="same historical Direct RGB/decoder/head; new branches seed42; fresh AdamW",
                 continuation_policy="preserve100epoch scheduler and optimizer; no schedule restart")
        result[mode] = c
    return result


def prepare():
    path = BATCH / "plan.json"
    if path.exists():
        plan = read(path)
        guard(plan, full=True)
        return plan
    from launch_material_categorical_20261020 import verify
    old = read(OLD_BATCH / "plan.json")
    verify(old, True)
    checks = read(BATCH / "engineering_verification.json")
    assert checks["status"] == "passed" and checks["cpu_tests_passed"]
    assert set(checks["cuda"]) == set(MODES)
    for row in checks["cuda"].values():
        assert row["restore_exact"] and row["real_crop"] == 512 and row["batch"] == 2
    for name in WORKFLOW:
        assert checks["workflow_sha256"]["work/" + name] == digest(ROOT / "work" / name)
    assert checks["source_sha256"] == source_hashes(ROOT)
    cs = configs()
    manifest = read(cs["N2"]["manifest"])
    ids = Path(manifest["splits"]["val"]["snapshot_path"]).read_text().split()
    masks = {str(Path(manifest["dataset_root"]) / manifest["mask_pattern"].format(id=ident)):
             digest(Path(manifest["dataset_root"]) / manifest["mask_pattern"].format(id=ident))
             for ident in ids}
    assert len(masks) == 25
    protected = {}
    for run in old["runs"].values():
        for name in ("checkpoints/best.pt", "checkpoints/last.pt", "config.json",
                     "status.json", "best_val_metrics.json", "best_val_confusion.npz",
                     "metrics_epochs.csv", "metrics_steps.csv", "metrics_val.csv"):
            file = Path(run) / name
            protected[str(file.relative_to(ROOT))] = digest(file)
    for name in ("comparison_summary.json", "N2_seed3408.json", "N3_seed3408.json"):
        file = OLD_BATCH / name
        protected[str(file.relative_to(ROOT))] = digest(file)
    workflow = [ROOT / "work" / name for name in WORKFLOW]
    workflow += [BATCH / "README.md", BATCH / "engineering_verification.json"]
    plan = dict(
        batch_id=BATCH.name, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        authorization=dict(user_instruction="按照这个方案开始训练",
                           skipped="fixed diagnostic cohort and sampling sweep",
                           scope="N2prime40,N3prime40; pass frozen screen -> both100 and samebest3408; reportstop"),
        configs=cs,
        runs={mode: str(ROOT / f"outputs/P1/sampler_aligned/seed_42/{BATCH.name}_{mode.lower()}")
              for mode in MODES},
        source_sha256=source_hashes(ROOT),
        workflow_sha256={str(file.relative_to(ROOT)): digest(file) for file in workflow},
        parent_plan_sha256=digest(OLD_BATCH / "plan.json"),
        prior_sha256=old["prior_sha256"], prior_parameters=old["prior_parameters"],
        condition_metadata_sha256=old["condition_metadata_sha256"],
        validation_masks_sha256=masks, protected_parent_results_sha256=protected,
        screen=dict(epoch=40, local_miou_min=0.8393,
                    local_boundary_f1_min=0.5356, local_minus_global_miou_min=0.002,
                    selection="exact epoch40 full25val; all three checks required"),
        final_screen=dict(old["screening"]),
        budget=dict(training_seed=42, number_of_runs=2, stage1_epochs_per_run=40,
                    maximum_epochs_per_run=100, updates_at40=2400, updates_at100=6000,
                    shared_prior_historical_epochs=40, new_exposures_at40=4800,
                    new_exposures_at100=12000),
        scope="No new diagnostics/test/material/seed/module/boundary-loss/precision change; no automatic further batches",
    )
    atomic_json(path, plan)
    guard(plan, full=True)
    return plan


def worker(plan, action, mode, stop=40):
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", action,
           "--mode", mode, "--stop-epoch", str(stop)]
    log = BATCH / f"{action}_{mode}_{stop if action == 'train' else 3408}.log"
    with log.open("a") as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def screen(plan):
    metrics = {}
    for mode in MODES:
        run = Path(plan["runs"][mode])
        metrics[mode] = read(run / "validation_epoch40.json")
        audit_metrics(run / "validation_epoch40.npz", metrics[mode])
    limits = plan["screen"]
    checks = dict(
        local_miou=metrics["N3"]["miou_4class"] >= limits["local_miou_min"],
        local_boundary=metrics["N3"]["boundary_f1_foreground"] >= limits["local_boundary_f1_min"],
        local_over_global=(metrics["N3"]["miou_4class"] - metrics["N2"]["miou_4class"]
                           >= limits["local_minus_global_miou_min"]),
    )
    result = dict(epoch=40, metrics=metrics, thresholds=limits, checks=checks,
                  passed=all(checks.values()), plan_sha256=digest(BATCH / "plan.json"))
    atomic_json(BATCH / "screen40.json", result)
    return result


def report(plan, gate, complete):
    results = {}
    for mode in MODES:
        run = Path(plan["runs"][mode])
        row = read(run / "best_val_metrics.json")
        audit_metrics(run / "best_val_confusion.npz", row["metrics"])
        results[mode] = dict(main=row, status=read(run / "status.json"),
                             checkpoint_sha256=digest(run / "checkpoints/best.pt"))
        if complete:
            second = read(BATCH / f"{mode}_seed3408.json")
            assert second["checkpoint_sha256"] == results[mode]["checkpoint_sha256"]
            assert second["plan_sha256"] == digest(BATCH / "plan.json")
            audit_metrics(BATCH / f"{mode}_seed3408.npz", second["metrics"])
            results[mode]["samebest3408"] = second
    checks = None
    if complete:
        old = read(OLD_BATCH / "comparison_summary.json")["results"]
        n = results["N3"]["main"]["metrics"]
        global_ = results["N2"]["main"]["metrics"]
        base, deterministic = old["N0"]["3407"]["metrics"], old["N1"]["3407"]["metrics"]
        limits = plan["final_screen"]
        checks = dict(
            local_vs_N0=n["miou_4class"] - base["miou_4class"] >= limits["N3_minus_N0_minimum_miou"],
            local_vs_N1=n["miou_4class"] - deterministic["miou_4class"] >= limits["N3_minus_N1_minimum_miou"],
            local_vs_global=n["miou_4class"] - global_["miou_4class"] >= limits["N3_minus_N2_minimum_miou"],
            thin_mean=(n["iou_monolayer"] + n["iou_fewlayer"] - base["iou_monolayer"]
                       - base["iou_fewlayer"]) / 2 >= limits["mean_mono_few_gain_vs_N0_minimum"],
            thin_boundary_guard=all(n[k] - base[k] >= -limits["maximum_individual_thin_boundary_decline"]
                                    for k in ("iou_monolayer", "iou_fewlayer", "boundary_f1_foreground")),
            secondary_positive=all(results["N3"]["samebest3408"]["metrics"]["miou_4class"] > x["miou_4class"]
                                   for x in (base, deterministic, results["N2"]["samebest3408"]["metrics"])),
        )
    summary = dict(status="completed" if complete else "stopped_after_screen",
                   screen40=gate, results=results, final_checks=checks,
                   final_screening_passed=None if checks is None else all(checks.values()),
                   budget=plan["budget"], plan_sha256=digest(BATCH / "plan.json"),
                   stop="No further training or evaluation queue")
    atomic_json(BATCH / "comparison_summary.json", summary)
    lines = ["# 五步区间对齐训练结果", "",
             "40轮筛选：" + ("通过" if gate["passed"] else "未通过，按计划停止"),
             "", "| 组别 | best轮数 | mIoU/% | 前景边界F1/% |",
             "|---|---:|---:|---:|"]
    for mode in MODES:
        row = results[mode]["main"]
        lines.append(f"| {mode}′ | {row['epoch']} | {100*row['metrics']['miou_4class']:.4f} | "
                     f"{100*row['metrics']['boundary_f1_foreground']:.4f} |")
    if checks is not None:
        lines += ["", "100轮主筛选：" + ("通过" if all(checks.values()) else "未通过")]
    lines += ["", "共享历史Direct40轮；两组新训练，非旧N2/N3追加训练。训练seed42；3408仅同best推理复核。"]
    (BATCH / "final_analysis.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("train", "secondary"))
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--stop-epoch", type=int, choices=(40, 100), default=40)
    args = parser.parse_args()
    if args.worker:
        plan = read(BATCH / "plan.json")
        guard(plan)
        if args.worker == "train":
            train(plan["configs"][args.mode], plan, plan["runs"][args.mode], args.stop_epoch)
        else:
            secondary(plan["configs"][args.mode], plan, plan["runs"][args.mode])
        return
    BATCH.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "outputs/.experiments.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stage = "freezing"
    try:
        progress(BATCH / "status.json", "running", stage=stage)
        plan = prepare()
        for mode in MODES:
            stage = "training_" + mode + "_prime_to40"
            progress(BATCH / "status.json", "running", stage=stage)
            guard(plan, full=True)
            worker(plan, "train", mode, 40)
        stage = "screening40"
        progress(BATCH / "status.json", "running", stage=stage)
        gate = screen(plan)
        if gate["passed"]:
            for mode in MODES:
                stage = "training_" + mode + "_prime_to100"
                progress(BATCH / "status.json", "running", stage=stage)
                guard(plan, full=True)
                worker(plan, "train", mode, 100)
            for mode in MODES:
                stage = mode + "_prime_samebest3408"
                progress(BATCH / "status.json", "running", stage=stage)
                guard(plan, full=True)
                path = BATCH / f"{mode}_seed3408.json"
                reusable = False
                if path.exists():
                    try:
                        row = read(path)
                        reusable = (row["status"] == "completed"
                                    and row["plan_sha256"] == digest(BATCH / "plan.json")
                                    and row["checkpoint_sha256"] == digest(Path(plan["runs"][mode]) / "checkpoints/best.pt")
                                    and path.with_suffix(".npz").exists())
                    except (ValueError, KeyError):
                        pass
                if not reusable:
                    worker(plan, "secondary", mode)
        stage = "reporting"
        progress(BATCH / "status.json", "running", stage=stage)
        summary = report(plan, gate, gate["passed"])
        guard(plan, full=True)
        progress(BATCH / "status.json", summary["status"], stage="finished",
                 screen40_passed=gate["passed"],
                 final_screening_passed=summary["final_screening_passed"])
    except BaseException as exc:
        progress(BATCH / "status.json", "failed", stage=stage, error=str(exc))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt("Termination requested")
    signal.signal(signal.SIGTERM, terminate)
    main()

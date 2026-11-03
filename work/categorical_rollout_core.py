"""Isolated on-policy categorical-state training; historical batches are immutable."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conditional_diffusion import material_experiment as legacy
from conditional_diffusion.material_categorical import categorical, classification_loss, label_channels, normalize
from conditional_diffusion.material_conditions import ConditionDataset, digest
from conditional_diffusion.metrics_impl import summarize, summarize_boundaries
from conditional_diffusion.runtime import (
    ExperimentLogger, StorageGuard, atomic_checkpoint, restore_rng, rng_state,
    source_hashes, write_json as legacy_json,
)
from conditional_diffusion.static_experiment import configure, scheduler_for, synchronize

BATCH = ROOT / "outputs/launches/p1_categorical_rollout_20261021"
OLD_BATCH = ROOT / "outputs/launches/p1_material_categorical_20261020"
TIMES = (100, 75, 50, 25, 1)
EARLIER = (75, 50, 25, 1, 0)


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f".tmp.{os.getpid()}")
    legacy_json(pending, value)
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    pending = path.with_name(path.name + f".tmp.{os.getpid()}")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def progress(path, status, **values):
    atomic_json(path, dict(status=status, pid=os.getpid(),
                          updated_at_utc=datetime.now(timezone.utc).isoformat(), **values))


@torch.no_grad()
def rollout_states(model, chain, image, conditions, indices, uniforms, prefix=None, intervals=None):
    """Actual five-step sampler prefix; never accepts labels or retains a graph."""
    relation, prior = chain.relation(conditions, model.variant == "N3")
    if prefix is None:
        prefix, intervals = chain.inference_matrices(relation, prior, list(TIMES))
    observed = categorical(prior, uniforms[:, 0])
    calls = int(indices.sum())
    if not calls:
        return observed, calls
    was_training = model.training
    model.eval()
    try:
        features = model.encode_conditions(image, conditions)
        for stage in range(int(indices.max())):
            active = torch.nonzero(indices > stage, as_tuple=False).flatten()
            t, earlier = TIMES[stage], TIMES[stage + 1]
            logits = model.forward_with_rgb(
                tuple(f[active] for f in features),
                2 * label_channels(observed[active]) - 1,
                torch.full((len(active),), t, device=image.device, dtype=torch.long),
            )
            posterior = chain.posterior(
                logits.softmax(1).permute(0, 2, 3, 1), observed[active],
                prefix[earlier][active], prefix[t][active], intervals[(earlier, t)][active],
            )
            observed[active] = categorical(posterior, uniforms[active, stage + 1])
    finally:
        model.train(was_training)
    return observed, calls


@torch.no_grad()
def training_states(model, chain, image, conditions, target, config,
                    indices=None, selected=None, uniforms=None):
    batch, _, height, width = image.shape
    if indices is None:
        indices = torch.randint(5, (batch,), device=image.device)
    assert indices.shape == (batch,) and bool(((indices >= 0) & (indices < 5)).all())
    # Consume identical random draws in every arm. Membership is independent of t.
    if selected is None:
        order = torch.randperm(batch, device=image.device)
        selected = torch.zeros(batch, dtype=torch.bool, device=image.device)
        selected[order[:round(batch * config["rollout_fraction"])]] = True
    assert selected.shape == (batch,) and selected.dtype == torch.bool
    if uniforms is None:
        uniforms = torch.rand((batch, 5, height, width), device=image.device)
    assert uniforms.shape == (batch, 5, height, width)
    relation, prior = chain.relation(conditions, model.variant == "N3")
    prefix, intervals = chain.inference_matrices(relation, prior, list(TIMES))
    times = torch.tensor(TIMES, device=image.device)[indices]
    current = torch.stack([prefix[int(t)][n] for n, t in enumerate(times)])
    probability = current.gather(
        -2, target[..., None, None].expand(*target.shape, 1, 4)
    ).squeeze(-2)
    observed = categorical(probability, uniforms[:, 0])
    calls = 0
    if bool(selected.any()):
        members = torch.nonzero(selected, as_tuple=False).flatten()
        generated, calls = rollout_states(
            model, chain, image[members], conditions[members], indices[members], uniforms[members],
            {k: v[members] for k, v in prefix.items()},
            {k: v[members] for k, v in intervals.items()},
        )
        observed[members] = generated
    tallies = {f"t{t}_count": int((times == t).sum()) for t in TIMES}
    tallies.update({f"rollout_t{t}_count": int(((times == t) & selected).sum()) for t in TIMES})
    tallies.update(timestep_sum=int(times.sum()), rollout_count=int(selected.sum()),
                   rollout_nonterminal_count=int((selected & (indices > 0)).sum()),
                   teacher_sample_forward_count=calls)
    return observed, times, tallies


def objective(model, chain, image, conditions, target, config, indices=None,
              selected=None, uniforms=None):
    assert config["kl_weight"] == 0 and config["rollout_fraction"] in (0., .5)
    observed, times, tallies = training_states(
        model, chain, image, conditions, target, config, indices, selected, uniforms
    )
    assert not observed.requires_grad
    features = model.encode_conditions(image, conditions)
    output = model.forward_with_rgb(features, 2 * label_channels(observed) - 1, times)
    loss, terms = classification_loss(output, target)
    terms.update(loss_classification=loss, loss_total=loss)
    return loss, terms, tallies


def train_step(model, chain, optimizer, image, conditions, target, config, indices=None, selected=None, uniforms=None):
    optimizer.zero_grad(set_to_none=True)
    loss, terms, tallies = objective(model, chain, image, conditions, target, config, indices, selected, uniforms)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite categorical-rollout loss")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["grad_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return {key: float(value.detach()) for key, value in terms.items()}, float(grad), tallies


def guard(plan, full=False):
    assert source_hashes(ROOT) == plan["source_sha256"], "Historical source/config changed"
    for field in ("workflow_sha256", "condition_metadata_sha256", "validation_masks_sha256"):
        for path, expected in plan[field].items():
            assert digest(ROOT / path) == expected, path
    assert digest(plan["configs"]["A"]["prior_checkpoint"]) == plan["prior_sha256"]
    assert digest(OLD_BATCH / "plan.json") == plan["parent_plan_sha256"]
    if full:
        from sampler_aligned_core import guard as previous_guard
        previous_guard(read(ROOT / "outputs/launches/p1_sampler_aligned_20261021/plan.json"), full=True)
        for path, expected in plan["protected_parent_results_sha256"].items():
            assert digest(ROOT / path) == expected, path


def validate_checkpoint(state, config, plan):
    assert state["config"] == config
    assert state["source_sha256"] == plan["source_sha256"]
    assert state["prior_sha256"] == plan["prior_sha256"]
    assert state["condition_metadata_sha256"] == plan["condition_metadata_sha256"]
    assert state["plan_sha256"] == digest(BATCH / "plan.json")


def metadata(config, plan):
    return dict(config=config, source_sha256=plan["source_sha256"],
                prior_sha256=plan["prior_sha256"],
                condition_metadata_sha256=plan["condition_metadata_sha256"],
                plan_sha256=digest(BATCH / "plan.json"))


def restore_best(run, state):
    atomic_npz(run / "best_val_confusion.npz", **state["validation_matrices"])
    atomic_json(run / "best_val_metrics.json",
                dict(epoch=state["epoch"], metrics=state["validation_metrics"]))


def audit_metrics(path, metrics):
    with np.load(path) as data:
        assert len(data["ids"]) == 25
        assert np.array_equal(data["per_image"].sum(0), data["global_matrix"])
        values = {**summarize(data["global_matrix"]),
                  **summarize_boundaries(data["boundary_counts"])}
        assert all(abs(value - metrics[key]) < 1e-12 for key, value in values.items())


def train(config, plan, run, stop_epoch):
    assert stop_epoch in (40, 100) and config["epochs"] == 100
    assert config["seed"] == 42 and config["batch_size"] == 2
    assert config["training_timesteps"] == list(TIMES)
    configure(config)
    guard(plan)
    run = Path(run)
    if run.exists() and not (run / "checkpoints/last.pt").exists():
        assert not (run / "metrics_steps.csv").exists()
        assert not (run / "checkpoints/best.pt").exists()
        run.rename(run.with_name(run.name + "_incomplete_" + str(time.time_ns())))
    resume = (run / "checkpoints/last.pt").exists()
    model, chain = legacy.build(config), legacy.chain_for(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = scheduler_for(optimizer, config)
    storage = StorageGuard(ROOT, run, config["storage"])
    storage.check(160 * 1024**2)
    epoch0 = step = 0
    best = -float("inf")
    last_validation = None
    if resume:
        state = torch.load(run / "checkpoints/last.pt", map_location="cpu", weights_only=False)
        validate_checkpoint(state, config, plan)
        for name in ("last.pt.tmp", "best.pt.tmp"):
            pending = run / "checkpoints" / name
            if pending.exists():
                pending.rename(pending.with_name(name + ".uncommitted_" + str(time.time_ns())))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        epoch0, step, best = state["epoch"], state["global_step"], state["best_miou"]
        last_validation = state["last_validation"]
        if (run / "checkpoints/best.pt").exists():
            selected = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
            validate_checkpoint(selected, config, plan)
            restore_best(run, selected)
            best = max(best, selected["validation_metrics"]["miou_4class"])
        if last_validation is not None:
            e = last_validation["epoch"]
            atomic_json(run / f"validation_epoch{e}.json", last_validation["metrics"])
            atomic_npz(run / f"validation_epoch{e}.npz", **last_validation["matrices"])
        del state
    else:
        run.mkdir(parents=True, exist_ok=False)
        (run / "checkpoints").mkdir()
        atomic_json(run / "config.json", config)
        parameters = sum(p.numel() for p in model.parameters())
        atomic_json(run / "run_metadata.json", dict(
            **metadata(config, plan), parameter_count=parameters,
            full_system_parameters=parameters + plan["prior_parameters"],
            budget="shared historical Direct40 + new40; conditional continuation to new100",
            train_records=120, train_unique_rgb=119, training_seed=42,
            planned_screen_updates=2400, maximum_updates=6000,
            precision="FP32; deterministic; matmulTF32false/cuDNNTF32true",
            torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
        ))

    def save_last(epoch):
        atomic_checkpoint(run / "checkpoints/last.pt", dict(
            **metadata(config, plan), model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            rng=rng_state(), epoch=epoch, global_step=step, best_miou=best,
            last_validation=last_validation,
        ), storage)

    if not resume:
        save_last(0)
    if epoch0 >= stop_epoch:
        progress(run / "status.json", "completed" if epoch0 == 100 else "screen_complete",
                 last_completed_epoch=epoch0, global_step=step)
        return
    dataset = ConditionDataset(config["manifest"], config["condition_cache"],
                               "train", crop=512, seed=config["seed"])
    assert len(dataset) == 120
    logger = ExperimentLogger(run, storage, step if resume else None, epoch0 if resume else None)
    committed = epoch0
    try:
        for epoch in range(epoch0 + 1, stop_epoch + 1):
            guard(plan)
            model.train()
            synchronize(config["device"])
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            seen, sums, tallies = 0, {}, {}
            lr = optimizer.param_groups[0]["lr"]
            progress(run / "status.json", "running", phase="training", current_epoch=epoch,
                     last_completed_epoch=committed, global_step=step)
            for image, conditions, target, _ in legacy.loader(dataset, config, epoch):
                image, conditions, target = [x.to(config["device"]) for x in (image, conditions, target)]
                terms, grad, counts = train_step(
                    model, chain, optimizer, image, conditions, target, config
                )
                assert counts["rollout_count"] == int(2 * config["rollout_fraction"])
                n = len(image)
                step += 1
                seen += n
                for key, value in terms.items():
                    sums[key] = sums.get(key, 0.) + n * value
                for key, value in counts.items():
                    tallies[key] = tallies.get(key, 0) + value
                if step % config["log_every"] == 0:
                    logger.scalars("steps", {"train_step/" + k: v for k, v in
                                   dict(**terms, grad_norm_pre_clip=grad, lr=lr).items()}, epoch, step)
                    progress(run / "status.json", "running", phase="training", current_epoch=epoch,
                             last_completed_epoch=committed, global_step=step)
            synchronize(config["device"])
            seconds = time.perf_counter() - start
            assert seen == 120 and step == epoch * 60
            values = {"train_epoch/" + k: v / seen for k, v in sums.items()}
            values.update({"categorical_train/" + k: v / seen for k, v in tallies.items()})
            values.update({
                "resources/train_seconds": seconds,
                "budget/teacher_sample_forwards": tallies.get("teacher_sample_forward_count", 0),
                "budget/rollout_exposures": tallies.get("rollout_count", 0),
                "resources/train_samples_per_second": seen / seconds,
                "resources/train_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "resources/train_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "optim/epoch_lr_group_0": lr,
                "budget/epoch_exposures": seen, "budget/optimizer_updates": 60,
            })
            logger.scalars("epochs", values, epoch, step)
            metrics = None
            if epoch % 20 == 0:
                progress(run / "status.json", "running", phase="validation", current_epoch=epoch,
                         last_completed_epoch=committed, global_step=step)
                metrics, matrices, previews, timing = legacy.evaluate(model, chain, config)
                assert metrics["image_count"] == 25 and metrics["nfe_total"] == 4375
                logger.scalars("val", {"val/" + k: v for k, v in metrics.items()}, epoch, step)
                logger.images(previews, epoch)
                last_validation = dict(epoch=epoch, metrics=metrics, matrices=matrices)
                atomic_json(run / f"validation_epoch{epoch}.json", metrics)
                atomic_npz(run / f"validation_epoch{epoch}.npz", **matrices)
                atomic_json(run / f"validation_timing_epoch{epoch}.json", timing)
                if metrics["miou_4class"] > best:
                    best = metrics["miou_4class"]
                    selected = dict(**metadata(config, plan), model=model.state_dict(),
                                    epoch=epoch, global_step=step,
                                    validation_metrics=metrics, validation_matrices=matrices)
                    atomic_checkpoint(run / "checkpoints/best.pt", selected, storage)
                    restore_best(run, selected)
            scheduler.step()
            logger.flush()
            save_last(epoch)
            committed = epoch
            status = "completed" if epoch == 100 else ("screen_complete" if epoch == stop_epoch else "running")
            progress(run / "status.json", status, phase="epoch_committed",
                     last_completed_epoch=epoch, global_step=step)
            print(dict(mode=config["display_mode"], epoch=epoch, global_step=step,
                       loss_total=sums["loss_total"] / seen, train_seconds=seconds,
                       val_miou=None if metrics is None else metrics["miou_4class"]), flush=True)
    except BaseException as exc:
        progress(run / "status.json", "interrupted", error=str(exc),
                 last_completed_epoch=committed, global_step=step,
                 recovery="resume from last.pt; replay any uncommitted epoch")
        raise
    finally:
        logger.close()


def secondary(config, plan, run):
    guard(plan)
    configure(config)
    run = Path(run)
    state = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    validate_checkpoint(state, config, plan)
    model = legacy.build(config)
    model.load_state_dict(state["model"])
    evaluation = dict(config, sampling_seed=3408)
    metrics, matrices, _, timing = legacy.evaluate(model, legacy.chain_for(config), evaluation)
    path = BATCH / f"{config['arm']}_seed3408.json"
    atomic_npz(path.with_suffix(".npz"), **matrices)
    atomic_json(BATCH / f"{config['arm']}_seed3408_timing.json", timing)
    result = dict(status="completed", epoch=state["epoch"], sampling_seed=3408,
                  metrics=metrics, checkpoint_sha256=digest(run / "checkpoints/best.pt"),
                  plan_sha256=digest(BATCH / "plan.json"), purpose="same3407best; no reselection")
    atomic_json(path, result)
    audit_metrics(path.with_suffix(".npz"), metrics)
    return result

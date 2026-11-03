"""Engineering checks only; isolated weights never enter effect training."""
import copy
import fcntl
import time
from pathlib import Path

import numpy as np
import torch

from categorical_rollout_core import (
    ROOT, BATCH, TIMES, atomic_json, atomic_checkpoint, digest, legacy, objective,
    read, restore_rng, rng_state, scheduler_for, source_hashes, StorageGuard,
    train_step, configure, rollout_states, training_states,
)
from conditional_diffusion.material_categorical import sample, categorical, classification_loss, label_channels
from conditional_diffusion.material_conditions import ConditionDataset
from launch_categorical_rollout_20261021 import WORKFLOW, MODES, configs, screen_checks
from verify_sampler_aligned_20261021 import cpu_tree, equal, fixture


def trajectory_test():
    """Compare all five states against unchanged production sampler, both chains."""
    for arm in ("B", "C"):
        c = dict(configs()[arm], device="cpu", cpu_threads=2)
        configure(c)
        model, chain = legacy.build(c), legacy.chain_for(c)
        model.eval()
        image, conditions, _ = fixture(1, 16)
        uniforms = torch.rand(1, 5, 16, 16)
        captured = []
        original = model.forward_with_rgb
        def record(features, state, t):
            captured.append((int(t[0]), state.argmax(1).clone()))
            return original(features, state, t)
        model.forward_with_rgb = record
        sample(model, chain, image, conditions, uniforms=uniforms, steps=5)
        del model.forward_with_rgb
        assert [t for t, _ in captured] == list(TIMES)
        for index in range(5):
            model.train()
            observed, calls = rollout_states(model, chain, image, conditions,
                                             torch.tensor([index]), uniforms)
            assert model.training and not observed.requires_grad
            assert calls == index and torch.equal(observed, captured[index][1]), (arm, index)
        _, prior = chain.relation(conditions, arm == "C")
        assert torch.equal(captured[0][1], categorical(prior, uniforms[:, 0]))
    print("PASS all five rollout states match unchanged sampler; terminal prior; model mode restored", flush=True)


def loss_and_initialization_test():
    cs = configs()
    reference = None
    for arm in MODES:
        c = dict(cs[arm], device="cpu", cpu_threads=2)
        configure(c)
        model, chain = legacy.build(c), legacy.chain_for(c)
        state = cpu_tree(model.state_dict())
        if reference is None:
            reference = state
        else:
            assert equal(reference, state), arm
        image, conditions, target = fixture()
        indices = torch.tensor([4, 2])
        selected = torch.tensor([True, False]) if arm != "A" else torch.tensor([False, False])
        uniforms = torch.rand(2, 5, 16, 16)
        observed, times, counts = training_states(model, chain, image, conditions, target,
                                                 c, indices, selected, uniforms)
        changed, _, _ = training_states(model, chain, image, conditions, (target+1)%4,
                                        c, indices, selected, uniforms)
        if arm != "A":
            assert torch.equal(observed[selected], changed[selected]), "Rollout state leaked GT"
        assert not observed.requires_grad and counts["rollout_count"] == int(selected.sum())
        assert counts["teacher_sample_forward_count"] == int(indices[selected].sum())
        forward_states = []
        def hook(_module, _args, output):
            forward_states.append((torch.is_grad_enabled(), output.requires_grad))
        handle = model.head.register_forward_hook(hook)
        loss, terms, _ = objective(model, chain, image, conditions, target, c,
                                    indices, selected, uniforms)
        handle.remove()
        assert forward_states == [(False, False)]*int(indices[selected].sum()) + [(True, True)]
        manual, _ = classification_loss(model.forward_with_rgb(
            model.encode_conditions(image, conditions), 2*label_channels(observed)-1, times), target)
        torch.testing.assert_close(loss, manual, rtol=0, atol=0)
        assert set(terms) == {"loss_ce", "loss_dice_fg", "loss_classification", "loss_total"}
        del loss, terms, manual
        opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        train_step(model, chain, opt, image, conditions, target, c, indices, selected, uniforms)
        for family in ("rgb_encoder", "decoder", "head", "condition_projection",
                       "state_encoder", "state_projection", "time_mlp"):
            assert sum(float(p.grad.abs().sum()) for name, p in model.named_parameters()
                       if name.startswith(family) and p.grad is not None) > 0, (arm, family)
    # Equal random-stream consumption / timestep draws across A/B/C is deliberate.
    streams = []
    for arm in MODES:
        c = dict(cs[arm], device="cpu", cpu_threads=2)
        configure(c)
        model, chain = legacy.build(c), legacy.chain_for(c)
        image, conditions, target = fixture()
        _, times, counts = training_states(model, chain, image, conditions, target, c)
        assert counts["rollout_count"] == (0 if arm == "A" else 1)
        streams.append((times, rng_state()))
    assert all(equal(streams[0], stream) for stream in streams[1:])
    print("PASS matching initialization/random draws; GT isolation; detached teacher; CE/Dice and branch gradients", flush=True)


def gate_test():
    limits = dict(local_miou_min=.8393, local_boundary_f1_min=.5456,
                  local_minus_global_miou_min=.002, local_minus_control_miou_min=.003)
    metrics = dict(A=dict(miou_4class=.83), B=dict(miou_4class=.835),
                   C=dict(miou_4class=.84, boundary_f1_foreground=.55))
    assert all(screen_checks(metrics, limits).values())
    for arm, key, value, check in (
        ("A", "miou_4class", .839, "local_over_control"),
        ("B", "miou_4class", .839, "local_over_global"),
        ("C", "miou_4class", .839, "local_miou"),
        ("C", "boundary_f1_foreground", .5455, "local_boundary"),
    ):
        altered = copy.deepcopy(metrics)
        altered[arm][key] = value
        assert not screen_checks(altered, limits)[check]
    print("PASS independent four-condition screening gate", flush=True)


def restore_test(mode, device):
    c = dict(configs()[mode], device=device, cpu_threads=2 if device == "cpu" else 4)
    configure(c)
    if device == "cuda":
        dataset = ConditionDataset(c["manifest"], c["condition_cache"], "train", 512, seed=42)
        dataset.epoch = 1
        items = [dataset[i] for i in range(2)]
        image, conditions, target = [torch.stack([row[k] for row in items]).to(device) for k in (0, 1, 2)]
        torch.cuda.reset_peak_memory_stats()
    else:
        image, conditions, target = fixture()
    model, chain = legacy.build(c), legacy.chain_for(c)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = scheduler_for(opt, c)
    start = time.perf_counter()
    first = train_step(model, chain, opt, image, conditions, target, c,
                       torch.tensor([4, 4], device=device))
    sched.step()
    payload = cpu_tree(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                            scheduler=sched.state_dict(), rng=rng_state()))
    run = BATCH / "engineering" / (device + "_" + mode)
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "last.pt"
    storage = StorageGuard(ROOT, run, c["storage"])
    atomic_checkpoint(checkpoint, payload, storage)
    expected = train_step(model, chain, opt, image, conditions, target, c)
    sched.step()
    expected_state = cpu_tree(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                                  scheduler=sched.state_dict(), rng=rng_state()))
    del model, opt, sched, payload
    if device == "cuda":
        torch.cuda.empty_cache()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = legacy.build(c)
    opt2 = torch.optim.AdamW(restored.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched2 = scheduler_for(opt2, c)
    restored.load_state_dict(saved["model"])
    opt2.load_state_dict(saved["optimizer"])
    sched2.load_state_dict(saved["scheduler"])
    restore_rng(saved["rng"])
    actual = train_step(restored, chain, opt2, image, conditions, target, c)
    sched2.step()
    state2 = cpu_tree(dict(model=restored.state_dict(), optimizer=opt2.state_dict(),
                          scheduler=sched2.state_dict(), rng=rng_state()))
    assert actual == expected, (mode, device, "step result")
    assert equal(state2, expected_state), (mode, device, "full state")
    result = dict(restore_exact=True, real_crop=image.shape[-1], batch=2,
                  initial_terms=first[0], seconds=time.perf_counter() - start,
                  checkpoint_sha256=digest(checkpoint))
    if device == "cuda":
        result.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                      total_memory_mib=torch.cuda.get_device_properties(0).total_memory / 1024**2)
        assert result["peak_reserved_mib"] < .9 * result["total_memory_mib"]
    print(f"PASS {device} {mode}: exact model/Adam/scheduler/RNG recovery", flush=True)
    return result



def main():
    BATCH.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "outputs/.experiments.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (BATCH / "plan.json").exists(), "Never overwrite a frozen engineering report"
    torch.set_num_interop_threads(1)
    trajectory_test()
    loss_and_initialization_test()
    gate_test()
    cpu = {arm: restore_test(arm, "cpu") for arm in MODES}
    assert torch.cuda.is_available()
    cuda = {arm: restore_test(arm, "cuda") for arm in MODES}
    result = dict(status="passed", cpu_tests_passed=True, cpu=cpu, cuda=cuda,
                  sampler_prefix_matches=True, no_gt_in_rollout=True, teacher_detached=True,
                  four_condition_gate_passed=True,
                  source_sha256=source_hashes(ROOT),
                  workflow_sha256={"work/" + name: digest(ROOT / "work" / name) for name in WORKFLOW},
                  scope="engineering only; weights discarded; no diagnostic/full-validation/effect-training")
    atomic_json(BATCH / "engineering_verification.json", result)
    print("All required engineering checks passed", flush=True)


if __name__ == "__main__":
    main()

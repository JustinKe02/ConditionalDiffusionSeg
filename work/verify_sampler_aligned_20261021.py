"""Necessary math/gradient/checkpoint/CUDA checks; no diagnostic or effect runs."""
import copy
import fcntl
import time
from pathlib import Path

import numpy as np
import torch

from sampler_aligned_core import (
    ROOT, BATCH, TIMES, EARLIER, atomic_json, atomic_checkpoint, digest,
    interval_matrices, legacy, objective, read, restore_rng, rng_state,
    scheduler_for, source_hashes, StorageGuard, train_step, configure,
)
from conditional_diffusion.material_categorical import MaterialModel
from conditional_diffusion.material_conditions import ConditionDataset
from launch_sampler_aligned_20261021 import WORKFLOW, configs


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def fixture(batch=2, side=16):
    image = torch.randn(batch, 3, side, side)
    conditions = torch.randn(batch, 8, side, side) * .1
    conditions[:, 3] = torch.rand(batch, side, side)
    conditions[:, 4:] = torch.randn(batch, 4, side, side).softmax(1)
    target = torch.randint(4, (batch, side, side))
    return image, conditions, target


def math_test():
    config = dict(configs()["N3"], device="cpu", cpu_threads=2)
    configure(config)
    chain = legacy.chain_for(config)
    _, conditions, target = fixture(5, 2)
    relation, prior = chain.relation(conditions, True)
    times, earlier = torch.tensor(TIMES), torch.tensor(EARLIER)
    previous, current, interval = interval_matrices(chain, relation, prior, times, earlier)
    # Independent double-precision products, without masked batched accumulation.
    for index, (t, s) in enumerate(zip(TIMES, EARLIER)):
        r, p = relation[index].double(), prior[index].double()
        a = torch.eye(4, dtype=torch.float64).expand(2, 2, 4, 4).clone()
        prefix = {0: a.clone()}
        segment = a.clone()
        for k in range(1, t + 1):
            beta, lam = 1 / (101 - k), 1 - k / 100
            q = (1 - beta) * torch.eye(4, dtype=torch.float64) + beta * (lam * r + (1 - lam) * p[..., None, :])
            a = a @ q
            prefix[k] = a.clone()
            if k > s:
                segment = segment @ q
        for actual, expected in ((previous[index], prefix[s]), (current[index], prefix[t]),
                                 (interval[index], segment)):
            torch.testing.assert_close(actual.double(), expected, atol=6e-6, rtol=1e-5)
    torch.testing.assert_close(current[0], prior[0, ..., None, :].expand_as(current[0]),
                               atol=2e-6, rtol=0)
    observed = target
    clean = torch.randn(5, 2, 2, 4).softmax(-1)
    actual = chain.posterior(clean, observed, previous, current, interval)
    reference = torch.zeros_like(actual)
    for n in range(5):
        for y in range(2):
            for x in range(2):
                j = observed[n, y, x]
                for k in range(4):
                    v = previous[n, y, x, k] * interval[n, y, x, :, j]
                    reference[n, y, x] += clean[n, y, x, k] * v / v.sum()
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=1e-5)
    print("PASS ordered interval products / terminal prior / exact Bayes mixture", flush=True)


def loss_and_initialization_test():
    configs_ = configs()
    model2 = legacy.build(dict(configs_["N2"], device="cpu"))
    model3 = legacy.build(dict(configs_["N3"], device="cpu"))
    assert equal(cpu_tree(model2.state_dict()), cpu_tree(model3.state_dict()))
    del model2, model3
    for mode in ("N2", "N3"):
        c = dict(configs_[mode], device="cpu", cpu_threads=2)
        configure(c)
        model, chain = legacy.build(c), legacy.chain_for(c)
        image, conditions, target = fixture()
        loss, terms, counts = objective(model, chain, image, conditions, target, c, torch.tensor([4, 4]))
        assert terms["loss_posterior_kl"].item() == 0
        assert torch.equal(loss, terms["loss_classification"])
        assert counts["t1_count"] == 2 and counts["t100_count"] == 0
        del loss, terms
        opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        train_step(model, chain, opt, image, conditions, target, c, torch.tensor([0, 1]))
        for family in ("rgb_encoder", "decoder", "head", "condition_projection",
                       "state_encoder", "state_projection", "time_mlp"):
            assert sum(float(p.grad.abs().sum()) for name, p in model.named_parameters()
                       if name.startswith(family) and p.grad is not None) > 0, (mode, family)
    print("PASS matched initialization / no duplicate t1 KL / branch gradients", flush=True)


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
                       torch.tensor([0, 4], device=device))
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
    math_test()
    loss_and_initialization_test()
    cpu = {mode: restore_test(mode, "cpu") for mode in ("N2", "N3")}
    assert torch.cuda.is_available()
    cuda = {mode: restore_test(mode, "cuda") for mode in ("N2", "N3")}
    result = dict(status="passed", cpu_tests_passed=True, cpu=cpu, cuda=cuda,
                  source_sha256=source_hashes(ROOT),
                  workflow_sha256={"work/" + name: digest(ROOT / "work" / name) for name in WORKFLOW},
                  scope="necessary engineering only; weights discarded; no diagnostic cohort/full validation/effect training")
    atomic_json(BATCH / "engineering_verification.json", result)
    print("All required engineering checks passed", flush=True)


if __name__ == "__main__":
    main()

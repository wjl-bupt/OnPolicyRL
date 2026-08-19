"""Experiment orchestration tests: job expansion, scheduling policy, GPU selection."""

import pytest

from oprl.experiment import GPUInfo, GPUScheduler, Sweep, list_sweeps, load_sweep


def test_job_expansion_env_outer_seed_inner():
    """Expansion must be env-outer, seed-inner -- that defines "environments serially"."""
    s = Sweep(algo="ppo", envs=["A-v0", "B-v0"], seeds=[1, 2])
    jobs = s.jobs()
    assert len(jobs) == 4
    assert [j["env"] for j in jobs] == ["A-v0", "A-v0", "B-v0", "B-v0"]
    assert [j["seed"] for j in jobs] == [1, 2, 1, 2]


def test_per_env_overrides():
    s = Sweep(envs=["A-v0", "B-v0"], seeds=[1],
              overrides={"lr": 1e-4}, per_env={"B-v0": {"total_steps": 999}})
    a, b = s.jobs()
    assert a["overrides"] == {"lr": 1e-4, "seed": 1}
    assert b["overrides"]["total_steps"] == 999 and b["overrides"]["lr"] == 1e-4


def test_shipped_sweeps_load():
    sw = list_sweeps()
    assert sw, "config/experiments.yaml should exist"
    for name in sw:
        assert load_sweep(name).jobs()


def test_sweep_rejects_typo():
    with pytest.raises(ValueError, match="unknown sweep options"):
        Sweep.from_dict({"algo": "ppo", "seads": [1]})


def test_unknown_sweep_lists_alternatives():
    with pytest.raises(KeyError, match="available"):
        load_sweep("no_such_sweep")


# ---------------- adaptive GPU selection ----------------


def test_scheduler_picks_gpu_with_most_free():
    s = GPUScheduler(per_run_mb=1000)
    s.gpus = [GPUInfo(0, 8000, 2000), GPUInfo(1, 8000, 6000)]
    assert s.acquire() == 1          # picks the card with the most free memory
    assert s.acquire() == 1          # 6000-1000=5000 is still the most
    assert s.gpus[1].assigned == 2


def test_scheduler_refuses_when_full():
    """When nothing fits, return None so the caller queues, rather than risking an OOM."""
    s = GPUScheduler(per_run_mb=5000)
    s.gpus = [GPUInfo(0, 8000, 6000)]
    assert s.acquire() == 0
    assert s.acquire() is None       # only 1000 left, below 5000
    s.release(0)
    assert s.acquire() == 0          # allocatable again after release


def test_scheduler_max_per_gpu():
    s = GPUScheduler(per_run_mb=100, max_per_gpu=2)
    s.gpus = [GPUInfo(0, 80000, 80000)]
    assert s.acquire() == 0 and s.acquire() == 0
    assert s.acquire() is None       # memory suffices but the per-card cap is hit


def test_scheduler_falls_back_to_cpu():
    s = GPUScheduler()
    s.gpus = []
    assert s.acquire() == -1         # -1 means CPU

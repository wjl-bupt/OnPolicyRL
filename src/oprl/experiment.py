"""Experiment orchestration: environments serially, seeds in parallel, adaptive GPU choice.

Default policy::

    for env in envs:              # environments run one family at a time
        parallel(seeds)           # seeds of one environment run concurrently

`mode: serial` runs everything sequentially, which is what you want on a shared server.

Adaptive GPU selection: before each run, pick the card with the most projected free
memory that can still fit the run (projected = free - assigned * per_run_mb). If nothing
fits, wait instead of overcommitting into an OOM.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
#  GPU probing and allocation
# --------------------------------------------------------------------------- #


@dataclass
class GPUInfo:
    index: int
    total_mb: int
    free_mb: int
    assigned: int = 0        # runs this orchestrator has placed on the card

    def projected_free(self, per_run_mb: int) -> int:
        return self.free_mb - self.assigned * per_run_mb


def probe_gpus() -> list[GPUInfo]:
    """Query memory via nvidia-smi. An empty list (no card / no driver) means fall back
    to CPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            gpus.append(GPUInfo(int(parts[0]), int(parts[1]), int(parts[2])))
    return gpus


class GPUScheduler:
    """Place runs on GPUs, telling the caller to wait when nothing fits.

    This is not precise memory management -- PyTorch's caching allocator makes exact
    prediction unrealistic. It is **conservative admission control**: better to run one
    fewer job in parallel than to OOM the whole batch.
    """

    def __init__(
        self,
        per_run_mb: int = 2048,
        max_per_gpu: int | None = None,
        visible: list[int] | None = None,
    ):
        self.per_run_mb = per_run_mb
        self.max_per_gpu = max_per_gpu
        self.gpus = [g for g in probe_gpus() if visible is None or g.index in visible]

    @property
    def available(self) -> bool:
        return bool(self.gpus)

    def acquire(self) -> int | None:
        """Return the assigned GPU index, or None when nothing fits (caller should wait).

        Returns -1 when there is no GPU at all, meaning "use CPU".
        """
        if not self.gpus:
            return -1
        best, best_free = None, -1
        for g in self.gpus:
            if self.max_per_gpu is not None and g.assigned >= self.max_per_gpu:
                continue
            free = g.projected_free(self.per_run_mb)
            if free >= self.per_run_mb and free > best_free:
                best, best_free = g, free
        if best is None:
            return None
        best.assigned += 1
        return best.index

    def release(self, index: int) -> None:
        for g in self.gpus:
            if g.index == index:
                g.assigned = max(0, g.assigned - 1)

    def refresh(self) -> None:
        """Re-read actual free memory -- other users may share this machine."""
        fresh = {g.index: g.free_mb for g in probe_gpus()}
        for g in self.gpus:
            if g.index in fresh:
                g.free_mb = fresh[g.index]


# --------------------------------------------------------------------------- #
#  Experiment definition
# --------------------------------------------------------------------------- #


@dataclass
class Sweep:
    """One batch of experiments; fields map one-to-one onto config/experiments.yaml keys."""

    algo: str = "ppo"
    envs: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    config: str | None = None                 # preset name in config/<algo>.yaml
    overrides: dict[str, Any] = field(default_factory=dict)
    # Per-environment overrides, e.g. {"HalfCheetah-v5": {"total_steps": 3000000}}
    per_env: dict[str, dict] = field(default_factory=dict)

    mode: str = "env_serial_seed_parallel"    # or "serial"
    max_parallel: int = 4                     # cap on concurrent seeds
    per_run_mb: int = 2048                    # estimated memory per run
    max_per_gpu: int | None = None            # cap on runs per card
    gpus: list[int] | None = None             # None = every visible card
    out_dir: str = "runs"
    poll_seconds: float = 5.0
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Sweep:
        d = {k: v for k, v in d.items() if k != "note"} | (
            {"note": d["note"]} if "note" in d else {}
        )
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown sweep options {sorted(unknown)}; available: {sorted(known)}")
        return cls(**d)

    def jobs(self) -> list[dict]:
        """Expand into a run list. The order is the execution order: env outer, seed inner."""
        out = []
        for env_id in self.envs:
            for seed in self.seeds:
                ov = {**self.overrides, **self.per_env.get(env_id, {}), "seed": seed}
                name = f"{self.algo}-{env_id.replace('/', '_')}-seed{seed}"
                out.append({
                    "algo": self.algo, "env": env_id, "seed": seed,
                    "config": self.config, "overrides": ov,
                    "run_dir": str(Path(self.out_dir) / name),
                })
        return out


# --------------------------------------------------------------------------- #
#  Execution
# --------------------------------------------------------------------------- #


def _cmd(job: dict, device: str) -> list[str]:
    cmd = [sys.executable, "-m", "oprl.cli", "train", job["algo"],
           "--env", job["env"], "--device", device,
           "--run_dir", job["run_dir"]]
    if job["config"]:
        cmd += ["--config", job["config"]]
    for k, v in job["overrides"].items():
        cmd += [f"--{k}", str(v)]
    return cmd


def run_sweep(sweep: Sweep, dry_run: bool = False) -> int:
    """Run a batch of experiments. Returns the number of failed runs."""
    jobs = sweep.jobs()
    if dry_run:
        print(f"# {len(jobs)} runs (mode={sweep.mode})")
        for j in jobs:
            print("  " + " ".join(_cmd(j, "<device>")))
        return 0

    sched = GPUScheduler(sweep.per_run_mb, sweep.max_per_gpu, sweep.gpus)
    print(f"[oprl] {len(jobs)} runs | mode={sweep.mode} | "
          f"GPU={[g.index for g in sched.gpus] or 'CPU'}")

    if sweep.mode == "serial":
        return _run_serial(jobs, sched, sweep)
    return _run_env_serial_seed_parallel(jobs, sched, sweep)


def _launch(job: dict, sched: GPUScheduler, sweep: Sweep):
    """Acquire a card and start the subprocess. Returns None when nothing fits."""
    idx = sched.acquire()
    if idx is None:
        return None
    device = "cpu" if idx < 0 else f"cuda:{idx}"
    Path(job["run_dir"]).mkdir(parents=True, exist_ok=True)
    logf = open(Path(job["run_dir"]) / "stdout.log", "w")
    env = dict(os.environ)
    if idx >= 0:
        # Expose only this card to the child, so it cannot grab another one.
        env["CUDA_VISIBLE_DEVICES"] = str(idx)
        device = "cuda:0"
    proc = subprocess.Popen(_cmd(job, device), stdout=logf, stderr=subprocess.STDOUT,
                            env=env)
    print(f"  ▶ {Path(job['run_dir']).name} → "
          f"{'cpu' if idx < 0 else f'gpu{idx}'} (pid {proc.pid})")
    return proc, idx, logf


def _run_serial(jobs: list[dict], sched: GPUScheduler, sweep: Sweep) -> int:
    failed = 0
    for job in jobs:
        while True:
            got = _launch(job, sched, sweep)
            if got:
                break
            sched.refresh()
            time.sleep(sweep.poll_seconds)
        proc, idx, logf = got
        rc = proc.wait()
        logf.close()
        sched.release(idx)
        failed += rc != 0
        print(f"  {'✓' if rc == 0 else '✗'} {Path(job['run_dir']).name} (rc={rc})")
    return failed


def _run_env_serial_seed_parallel(
    jobs: list[dict], sched: GPUScheduler, sweep: Sweep
) -> int:
    """Group by env: parallel within a group, serial between groups (a group starts only
    after the previous one has fully finished)."""
    failed = 0
    groups: dict[str, list[dict]] = {}
    for j in jobs:
        groups.setdefault(j["env"], []).append(j)

    for env_id, group in groups.items():
        print(f"\n[oprl] === {env_id} ({len(group)} seeds in parallel) ===")
        pending, running = list(group), []
        while pending or running:
            # Fill the parallel slots as far as memory allows.
            while pending and len(running) < sweep.max_parallel:
                got = _launch(pending[0], sched, sweep)
                if got is None:      # not enough memory; wait for a running job to exit
                    break
                running.append((pending.pop(0), *got))
            if not running:
                sched.refresh()
                time.sleep(sweep.poll_seconds)
                continue
            time.sleep(sweep.poll_seconds)
            still = []
            for job, proc, idx, logf in running:
                rc = proc.poll()
                if rc is None:
                    still.append((job, proc, idx, logf))
                    continue
                logf.close()
                sched.release(idx)
                failed += rc != 0
                print(f"  {'✓' if rc == 0 else '✗'} "
                      f"{Path(job['run_dir']).name} (rc={rc})")
            running = still
    return failed


def load_sweep(name: str, path: str | Path | None = None) -> Sweep:
    """Load one sweep from config/experiments.yaml."""
    from .config import CONFIG_DIR, _read

    p = Path(path) if path else CONFIG_DIR / "experiments.yaml"
    doc = _read(p)
    if name not in doc:
        raise KeyError(
            f"{p} has no sweep {name!r}; available: {[k for k in doc if not k.startswith('_')]}"
        )
    return Sweep.from_dict(doc[name] or {})


def list_sweeps(path: str | Path | None = None) -> dict[str, str]:
    from .config import CONFIG_DIR, _read

    p = Path(path) if path else CONFIG_DIR / "experiments.yaml"
    if not Path(p).is_file():
        return {}
    return {
        k: str((v or {}).get("note", "")).strip()
        for k, v in _read(p).items()
        if not k.startswith("_")
    }


__all__ = [
    "Sweep", "run_sweep", "load_sweep", "list_sweeps",
    "GPUScheduler", "GPUInfo", "probe_gpus",
]

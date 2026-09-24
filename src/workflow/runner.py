"""Runner -- L4 stages (2) run + the (3) monitor / (4) report wiring.

`run(experiment_path)`:  load -> expand the grid -> run each Job sequentially.
`run_job(job)`: build this job's env / policy / logger / monitor, honor the
`resume` toggle, call the algorithm through `get_algo` (the only L2 seam), and
record provenance (config.json) + metrics (metrics.jsonl) + logs under the job's
run_dir. A crash in one job is caught, written to error.log, and does NOT abort
the sweep.

resume toggle (config field `resume`):
    resume=True  + job finished (DONE marker)      -> skip
    resume=True  + checkpoint.pt present, no DONE   -> load policy+opt+rng+step,
                                                       continue from there
    resume=False                                    -> clear old markers, retrain

This is the R2 vertical slice: a grid runs to completion sequentially. Parallel
scheduling and the results/plot three-piece are separate follow-ups.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python src/workflow/runner.py experiment.yaml`: put `src` on the path
# BEFORE the framework imports below resolve. Under pytest, conftest already did.
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import platform
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime

import torch as th

from architecture.actor_critic import PolicyConfig, build_policy
from common.checkpoint import (fingerprint, latest_checkpoint, load_checkpoint,
                               peek_checkpoint)
from common.env.adapter import GymVectorAdapter
from common.env.make_env import make_env
from common.logger import Logger
from workflow.experiment import Job, expand, load_experiment
from workflow.monitor import Monitor
from workflow.registry import get_algo
from workflow.report import report


# --------------------------------------------------------------------------- #
#  building blocks from one job's resolved config
# --------------------------------------------------------------------------- #


def _adapter(env_id: str, num_envs: int, seed: int, device: str) -> GymVectorAdapter:
    return GymVectorAdapter(make_env(env_id, num_envs=num_envs, seed=seed),
                            device=device)


def _algo_config(job: Job, config_cls: type, device: str | None):
    """`algo_params` -> the algorithm's config dataclass. `seed` is swept at the
    top level, so it is injected here, never read from algo_params."""
    params = dict(job.config.get("algo_params", {}))
    params["seed"] = job.seed
    if device is not None:
        params["device"] = device
    try:
        return config_cls(**params)
    except TypeError as e:
        raise ValueError(f"bad algo_params for job {job.label}: {e}") from e


def _policy_config(job: Job) -> PolicyConfig:
    """`network_params` -> PolicyConfig (plain fields only; `activation` /
    `optimizer_cls` are Python types and stay at their defaults). `net_arch`
    arrives from YAML as a list -> coerced to the tuple PolicyConfig expects."""
    params = dict(job.config.get("network_params", {}))
    if isinstance(params.get("net_arch"), list):
        params["net_arch"] = tuple(params["net_arch"])
    try:
        return PolicyConfig(**params)
    except TypeError as e:
        raise ValueError(f"bad network_params for job {job.label}: {e}") from e


def _dump_config(path: Path, job: Job, cfg, device: str) -> None:
    """Persist the fully-resolved job config for provenance (never raises)."""
    resolved = {
        "algo": job.algo, "env": job.env, "seed": job.seed, "device": device,
        "identity": job.identity, "label": job.label, "resume": job.resume,
        "config": job.config,
        "algo_config": asdict(cfg) if is_dataclass(cfg) else str(cfg),
    }
    try:
        path.write_text(json.dumps(resolved, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
#  one job
# --------------------------------------------------------------------------- #


def _hardware(device: str) -> dict:
    """A coarse hardware descriptor for the checkpoint fingerprint: enough to
    catch a torch upgrade or a device/arch change (which make old weights +
    optimizer state unsafe to resume), without being so fine that a trivial OS
    detail invalidates a pool -- device + torch version + machine arch."""
    return {"device": str(device), "torch": th.__version__,
            "machine": platform.machine()}


def _reset_run(done: Path, pool: Path) -> None:
    """Fresh start: drop the DONE marker and clear the model pool, so a rerun does
    not resume stale or incompatible weights."""
    done.unlink(missing_ok=True)
    if pool.is_dir():
        for p in pool.glob("model_step-*.pt"):
            p.unlink(missing_ok=True)


def run_job(job: Job, device: str | None = None) -> dict:
    run_dir = Path(job.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / "DONE"
    pool = run_dir / "model_pools"

    if job.resume and done.exists():
        return {"status": "skipped", "label": job.label,
                "run_dir": str(run_dir), "identity": job.identity}
    if not job.resume:                       # fresh start: drop stale markers + pool
        _reset_run(done, pool)

    entry = get_algo(job.algo)
    cfg = _algo_config(job, entry.config_cls, device)
    dev = getattr(cfg, "device", device or "cpu")
    _dump_config(run_dir / "config.json", job, cfg, dev)

    # fingerprint of THIS setup (resolved algo config + hardware) -- stamped into
    # every checkpoint's meta and checked before resuming into an old one.
    hw = _hardware(dev)
    fp = fingerprint(asdict(cfg) if is_dataclass(cfg) else {"config": str(cfg)}, hw)

    log = Logger(run_dir=run_dir)
    try:
        env = _adapter(job.env, cfg.num_envs, job.seed, dev)
        policy = build_policy(env.obs_space, env.action_space, _policy_config(job))
        policy.to(dev)                        # obs (adapter) + buffer live on `dev`;
        #                                       put the network there too (closes the
        #                                       device assembly -- CPU path is a no-op)
        resume_state = None
        if job.resume:
            latest = latest_checkpoint(pool)      # newest model_step-<N>.pt, or None
            if latest is not None:
                saved_fp = peek_checkpoint(latest).get("meta", {}).get("fingerprint")
                if saved_fp == fp:
                    resume_state = load_checkpoint(latest, policy)
                    log.info("resumed from checkpoint", ckpt=latest.name,
                             **{k: resume_state[k]
                                for k in ("global_step", "iteration")})
                else:
                    # config or hardware changed since the pool was written: the
                    # old weights are not safely resumable -> warn + retrain clean.
                    log.warning("checkpoint fingerprint mismatch -- retraining "
                                "from scratch", ckpt=latest.name)
                    _reset_run(done, pool)

        mon_kw = dict(job.config.get("monitor", {}))
        eval_seed = int(mon_kw.pop("eval_seed", job.seed + 10_000))
        monitor = Monitor(
            policy,
            env_factory=lambda: _adapter(job.env, 1, eval_seed, dev),
            log=log, run_dir=run_dir, total_steps=cfg.total_steps,
            eval_seed=eval_seed,
            meta={"identity": job.identity, "label": job.label,
                  "algo": job.algo, "env": job.env, "seed": job.seed,
                  "fingerprint": fp,
                  "timestamp": datetime.now().isoformat(timespec="seconds")},
            **mon_kw,
        )

        summary = entry.train(cfg, env, policy, log, monitor, resume=resume_state)
        env.close()
        done.write_text(json.dumps({"identity": job.identity, **summary}) + "\n",
                        encoding="utf-8")
        log.success("job complete", **summary)
        return {"status": "done", "label": job.label, "run_dir": str(run_dir),
                "identity": job.identity, **summary}
    except Exception:
        log.crash(traceback.format_exc())
        log.error("job crashed -- see error.log", label=job.label)
        return {"status": "failed", "label": job.label,
                "run_dir": str(run_dir), "identity": job.identity}
    finally:
        log.close()


# --------------------------------------------------------------------------- #
#  the sweep
# --------------------------------------------------------------------------- #


def _snapshot_config(src: Path, run_dir: Path, log: Logger) -> None:
    """Archive the launched experiment file verbatim under the run root, so the
    exact declaration that produced this sweep is preserved even if the source is
    edited afterwards. This is the RAW source; each job's config.json holds the
    resolved config. Best-effort: a snapshot failure warns but never aborts."""
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        dst = run_dir / f"experiment.snapshot{src.suffix.lower()}"
        dst.write_bytes(src.read_bytes())
        log.info("config snapshot written", path=str(dst))
    except Exception as e:  # noqa: BLE001
        log.warning(f"config snapshot failed: {e}")


def run(experiment_path: str | Path, device: str | None = None) -> list[dict]:
    """Load an experiment, expand its grid, and run every job sequentially.

    Jobs come out of `expand()` already ordered experiment -> env -> seed, so
    running them in order finishes every env x run of one experiment before the
    next experiment begins."""
    exp = load_experiment(experiment_path)
    jobs = expand(exp)
    log = Logger()                            # console-only orchestration narration
    _snapshot_config(Path(experiment_path), exp.run_dir, log)   # [1] launch snapshot
    log.info(f"experiment '{exp.name}'", algo=exp.algo,
             envs=",".join(exp.envs), runs=exp.runs,
             jobs=len(jobs), resume=exp.resume)

    summaries: list[dict] = []
    for i, job in enumerate(jobs, 1):
        log.info(f"[{i}/{len(jobs)}] {job.label}", run_dir=str(job.run_dir))
        summaries.append(run_job(job, device=device))

    counts = {s: sum(x["status"] == s for x in summaries)
              for s in ("done", "skipped", "failed")}
    try:
        report(exp, log=log)              # stage (4): the results three-piece
    except Exception as e:                # best-effort; the runs already happened
        log.warning(f"report generation failed: {e}")
    log.success(f"experiment '{exp.name}' finished", **counts)
    return summaries


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="oprl experiment runner (L4)")
    ap.add_argument("experiment", help="path to a .yaml / .json / .toml experiment file")
    ap.add_argument("--device", default=None, help="override the config's device")
    args = ap.parse_args()
    run(args.experiment, device=args.device)

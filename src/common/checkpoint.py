"""Checkpoint save / load -- a primitive (L1), used by the workflow's Monitor.

What a checkpoint holds (DESIGN_v2 §5.1), and why each piece:

* ``policy``     -- ``nn.Module.state_dict``; the weights, obviously.
* ``optimizer``  -- ``policy.optimizer.state_dict``; Adam's moment estimates. Drop
                    these and a resumed run restarts Adam cold, a visible bump in
                    the learning curve.
* ``torch_rng``  -- the global torch RNG state. Rollout collection samples actions
                    from the global generator, so restoring it keeps a resumed
                    run's collection on the same stream (the per-update shuffle
                    generator is re-seeded deterministically from ``iteration`` and
                    needs no saving).
* ``global_step`` / ``iteration`` -- where the outer loop was; the runner injects
                    these back so ``while global_step < total_steps`` continues.
* ``meta``       -- opaque provenance the caller wants stored (job identity, the
                    resolved config, ...); this module never interprets it.

The write is atomic (temp file + ``os.replace``): a process killed mid-save
leaves the previous good checkpoint intact rather than a truncated one. This
module is stdlib+torch only and imports nothing from the framework's upper layers.

Two helpers support the workflow's model pool (``run_dir/model_pools``):
``fingerprint`` decides whether a saved checkpoint is compatible with the current
run, and ``latest_checkpoint`` picks the newest ``model_step-<N>.pt`` to resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch as th


def save_checkpoint(path: str | Path, policy, *, global_step: int,
                    iteration: int, meta: dict | None = None) -> None:
    """Atomically write a checkpoint for ``policy`` at ``path``.

    ``policy`` must expose ``state_dict()`` and an ``optimizer`` attribute (the
    Policy contract from architecture/actor_critic.py). ``meta`` is stored
    verbatim and returned untouched by :func:`load_checkpoint`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opt = getattr(policy, "optimizer", None)
    payload: dict[str, Any] = {
        "policy": policy.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "torch_rng": th.get_rng_state(),          # ByteTensor on CPU
        "global_step": int(global_step),
        "iteration": int(iteration),
        "meta": dict(meta or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    th.save(payload, tmp)
    os.replace(tmp, path)                          # atomic on POSIX


def load_checkpoint(path: str | Path, policy, *,
                    map_location: str | th.device = "cpu") -> dict:
    """Restore ``policy`` (weights + optimizer) and the global RNG state from
    ``path``; return ``{global_step, iteration, meta}`` for the caller to inject
    back into the training loop.

    ``weights_only=False`` is required: the payload carries the RNG ByteTensor
    and a free-form ``meta`` dict, not just plain weight tensors (torch>=2.6
    defaults ``weights_only=True``, which would reject them).
    """
    ckpt = th.load(path, map_location=map_location, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    opt = getattr(policy, "optimizer", None)
    if opt is not None and ckpt.get("optimizer") is not None:
        opt.load_state_dict(ckpt["optimizer"])
    rng = ckpt.get("torch_rng")
    if rng is not None:
        # set_rng_state wants a CPU ByteTensor; map_location may have moved it
        th.set_rng_state(rng.to("cpu", th.uint8))
    return {"global_step": int(ckpt.get("global_step", 0)),
            "iteration": int(ckpt.get("iteration", 0)),
            "meta": ckpt.get("meta", {})}


def peek_checkpoint(path: str | Path,
                    map_location: str | th.device = "cpu") -> dict:
    """Read the progress markers without restoring into any policy -- for a
    runner deciding whether a job is finished. Loads the whole payload (cheap for
    on-policy nets) but returns only the scalars."""
    ckpt = th.load(path, map_location=map_location, weights_only=False)
    return {"global_step": int(ckpt.get("global_step", 0)),
            "iteration": int(ckpt.get("iteration", 0)),
            "meta": ckpt.get("meta", {})}


def fingerprint(algo_config: dict, hardware: dict) -> str:
    """A stable md5 over the algorithm config + a hardware descriptor -- the
    IDENTITY OF THE SETUP that produced a checkpoint, used to decide whether a
    saved model is safe to resume into the current run.

    Deliberately excludes any timestamp: two runs of the same config on the same
    hardware share a fingerprint (so one can resume the other), while a changed
    hyperparameter or a different torch/device/arch changes it (so the runner
    refuses to load incompatible weights and retrains from scratch instead).
    """
    blob = json.dumps({"config": algo_config, "hardware": hardware},
                      sort_keys=True, default=str)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


_STEP_RE = re.compile(r"model_step-(\d+)\.pt$")


def latest_checkpoint(model_pool_dir: str | Path) -> Path | None:
    """The ``model_step-<N>.pt`` with the largest N in ``model_pool_dir`` (the
    model pool), or None if the pool is empty or absent. Resume always continues
    from the newest step -- earlier checkpoints are kept for provenance/rollback,
    not loaded."""
    d = Path(model_pool_dir)
    if not d.is_dir():
        return None
    best: Path | None = None
    best_step = -1
    for p in d.glob("model_step-*.pt"):
        m = _STEP_RE.search(p.name)
        if m and int(m.group(1)) > best_step:
            best, best_step = p, int(m.group(1))
    return best

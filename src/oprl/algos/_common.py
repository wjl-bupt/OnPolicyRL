"""Mechanical setup shared by every algorithm's `train()`.

**Why this exists.** `vmpo.train()` was 85 lines, 51 of which were byte-identical to
`ppo.train()`: device resolution, seeding, logger, timer, estimator lookup, normalizers,
buffer construction, iteration count, lr annealing, and the logging tail. Copying those
51 lines is exactly the CleanRL failure mode DESIGN.md §1 rejects, just one size down --
and it had already diverged, with `vmpo` silently ignoring both config-declared buffer
fields and an estimator's `resolve_fields()`.

**What this deliberately is not.** Not a base class, and not a training loop. Every
function here is called explicitly from the algorithm file, so the loop stays readable
top to bottom (DESIGN.md §2, principle 5: no hidden state, no callbacks). The rule
applied: mechanical plumbing is shared, anything expressing the *update rule* stays in
the algorithm file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..advantages import get_estimator
from ..buffer import RolloutBuffer
from ..logger import Logger
from ..metrics import Timer, explained_variance
from ..norm import ObsNormalizer, RewardNormalizer
from ..seeding import seed_everything
from ..types import EnvAdapter, Policy


def buffer_extra(cfg, estimator, env) -> dict | None:
    """Extra buffer fields = config declarations + estimator requirements (§4.3)."""
    from ..schema import env_symbols, schema_from_config

    spec = (cfg.buffer or {}).get("extra") if isinstance(getattr(cfg, "buffer", None), dict) else None
    declared = schema_from_config(spec, env_symbols(env.obs_space, env.action_space))
    # Env-dependent field shapes take priority over the estimator's static declaration.
    from_est = dict(getattr(estimator, "extra_fields", None) or {})
    if hasattr(estimator, "resolve_fields"):
        from_est.update(estimator.resolve_fields(env.obs_space, env.action_space) or {})
    dup = set(declared) & set(from_est)
    if dup:
        raise ValueError(f"buffer.extra collides with estimator fields: {sorted(dup)}")
    return {**declared, **from_est} or None


@dataclass
class Runtime:
    """The objects every on-policy `train()` needs before its first iteration.

    A plain data holder -- it owns no loop logic, so an algorithm can use any subset and
    build the rest by hand.
    """

    cfg: Any
    env: EnvAdapter
    policy: Policy
    log: Logger
    timer: Timer
    estimator: Any
    device: torch.device
    buf: RolloutBuffer
    obs_norm: ObsNormalizer | None
    reward_norm: RewardNormalizer | None
    n_iters: int
    batch: int


def setup(
    cfg,
    env: EnvAdapter,
    policy: Policy,
    log: Logger | None = None,
    estimator=None,
) -> Runtime:
    """Build everything an on-policy training loop needs, in the order it must happen.

    Seeding comes first, before any parameter exists: `cfg.seed` must determine the whole
    run rather than just the environment (see oprl/seeding.py).
    """
    device = cfg.resolve_device()
    seed_everything(cfg.seed, cfg.deterministic)
    log = log or Logger(run_dir=cfg.run_dir)
    est = get_estimator(estimator or cfg.advantage, cfg)

    obs_norm = ObsNormalizer(env.obs_space.shape, device) if cfg.norm_obs else None
    reward_norm = (
        RewardNormalizer(env.num_envs, cfg.gamma, device) if cfg.norm_reward else None
    )

    buf = RolloutBuffer(
        cfg.rollout_len, env.num_envs, env.obs_space, env.action_space, device,
        extra=buffer_extra(cfg, est, env),
    )

    batch = cfg.rollout_len * env.num_envs
    return Runtime(
        cfg=cfg, env=env, policy=policy, log=log, timer=Timer(), estimator=est,
        device=device, buf=buf, obs_norm=obs_norm, reward_norm=reward_norm,
        n_iters=max(1, cfg.total_steps // batch), batch=batch,
    )


def begin_iteration(cfg, opt, it: int, n_iters: int) -> None:
    """Publish training progress and anneal the learning rate.

    `cfg._progress` is read by surrogates that anneal on it (MDPO). It is only set when
    the config declares the field, so a config without it is left untouched rather than
    growing a stray attribute.
    """
    progress = it / max(1, n_iters)
    if hasattr(cfg, "_progress"):
        cfg._progress = progress
    if cfg.anneal_lr:
        for g in opt.param_groups:
            g["lr"] = (1.0 - progress) * cfg.lr


def log_iteration(rt: Runtime, opt, stats: dict, steps: int) -> None:
    """The per-iteration logging tail: critic health, lr, throughput, timing."""
    if stats and "_value" in stats and "_returns" in stats:
        rt.log.record(
            "diag/explained_variance",
            explained_variance(stats["_value"], stats["_returns"]),
        )
    rt.log.record("train/lr", opt.param_groups[0]["lr"])
    elapsed = sum(rt.timer.acc.values())
    if elapsed > 0:
        rt.log.record("perf/sps", steps / elapsed)
    rt.log.add(**rt.timer.drain())


def finish(log: Logger, global_step: int) -> None:
    log.dump(global_step)
    log.close()


__all__ = ["Runtime", "setup", "begin_iteration", "log_iteration", "finish",
           "buffer_extra"]

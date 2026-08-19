"""PPO -- flat, linear, readable top to bottom, with no base class.

Every hyperparameter is listed explicitly in PPOConfig; none are hard-coded
(DESIGN.md §4.6). A2C is just a config of this algorithm:
num_epochs=1, num_minibatches=1, clip_coef=inf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from ..advantages import get_estimator
from ..buffer import RolloutBuffer
from ..config import Config
from ..logger import Logger
from ..metrics import Timer, explained_variance
from ..norm import ObsNormalizer, RewardNormalizer
from ..objectives import get_surrogate, get_value_loss
from ..rollout import collect
from ..types import EnvAdapter, Policy


@dataclass
class PPOConfig(Config):
    num_epochs: int = 10
    num_minibatches: int = 4
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: Literal["batch", "minibatch", "none"] = "minibatch"
    target_kl: float | None = None
    norm_obs: bool = False
    norm_reward: bool = False

    # --- Pluggable components (DESIGN.md §4.6 / §4.7) ---
    # Three accepted forms: a name, {name: ..., kwargs}, or {from: ./my.py:MyClass}
    surrogate: Any = "ppo"       # policy loss: ppo|tr_ppo|spo|dpo|mdpo|ppo_rpe|apo
    advantage: Any = "gae"       # advantage estimator: gae|vtrace_free_mc
    value_loss: Any = "clipped"  # value loss: clipped|mse|huber
    network: Any = None          # network: {hidden:[...], encoder:..., from:...}
    buffer: Any = None           # extra buffer fields: {extra: {name: {shape, dtype}}}

    # --- Per-surrogate hyperparameters, listed explicitly rather than hidden in kwargs ---
    rollback_alpha: float = 0.3      # TR-PPO
    spo_beta: float = 1.0            # SPO
    dpo_alpha: float = 2.0           # DPO
    dpo_beta: float = 0.6            # DPO
    mdpo_tk: float = 1.0             # MDPO
    rpe_alpha: float = 0.5           # PPO-RPE
    apo_uarr_coef: float = 0.1       # APO
    apo_resample: bool = True        # APO: resample unsampled actions for UARR

    # Written by train() each iteration, for surrogates that anneal (e.g. MDPO).
    _progress: float = 0.0
    _resampled_logratio: object = None


def buffer_extra(cfg: PPOConfig, estimator, env) -> dict:
    """Extra buffer fields = config declarations + estimator requirements (§4.3)."""
    from ..schema import env_symbols, schema_from_config

    spec = (cfg.buffer or {}).get("extra") if isinstance(cfg.buffer, dict) else None
    declared = schema_from_config(spec, env_symbols(env.obs_space, env.action_space))
    from_est = getattr(estimator, "extra_fields", None) or {}
    dup = set(declared) & set(from_est)
    if dup:
        raise ValueError(f"buffer.extra collides with estimator fields: {sorted(dup)}")
    return {**declared, **from_est} or None


def ppo_loss(
    policy: Policy, mb, cfg: PPOConfig, surrogate=None, estimator=None, value_loss=None
) -> tuple[torch.Tensor, dict]:
    """Policy loss (chosen by the surrogate) plus value loss. Short enough to read at once.

    There is exactly one call into the policy -- `evaluate()`. That single call site is
    what makes "swap the architecture" and "recurrent does not fork the code" true.
    """
    surrogate = surrogate or get_surrogate(cfg.surrogate)
    logp, entropy, value = policy.evaluate(mb.obs, mb["action"], valid=mb.get("valid"))
    logratio = logp - mb["logprob"]
    ratio = logratio.exp()

    adv = mb["advantages"]
    if cfg.norm_adv == "minibatch":
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # The single policy-objective entry point: swapping algorithms swaps this object.
    pg_loss, sur_stats = surrogate(ratio, logp, mb["logprob"], adv, cfg)

    returns = mb["returns"]
    value_loss = value_loss or get_value_loss(
        cfg.value_loss if cfg.clip_vloss else "mse"
    )
    v_loss, v_extra = value_loss(value, returns, mb["value"], cfg)
    sur_stats = {**sur_stats, **v_extra}

    # An estimator may take over the critic loss (DAE/RVL change exactly that target).
    if estimator is not None:
        custom = estimator.critic_loss(policy, mb, cfg)
        if custom is not None:
            v_loss, v_stats = custom
            sur_stats = {**sur_stats, **v_stats}

    ent = entropy.mean()
    loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()  # k3 estimator; steadier than -logratio

    return loss, {
        "loss/policy": pg_loss.item(),
        "loss/value": v_loss.item(),
        "loss/entropy": ent.item(),
        "loss/total": loss.item(),
        "diag/kl": approx_kl.item(),
        **sur_stats,
        "_kl": approx_kl.item(),
        "_value": value.detach(),
        "_returns": returns,
    }


def train(
    cfg: PPOConfig,
    env: EnvAdapter,
    policy: Policy,
    log: Logger | None = None,
    estimator=None,
) -> Policy:
    device = cfg.resolve_device()
    log = log or Logger(run_dir=cfg.run_dir)
    timer = Timer()
    estimator = get_estimator(estimator or cfg.advantage, cfg)
    surrogate = get_surrogate(cfg.surrogate)
    value_loss = get_value_loss(cfg.value_loss)

    obs_norm = (
        ObsNormalizer(env.obs_space.shape, device) if cfg.norm_obs else None
    )
    reward_norm = (
        RewardNormalizer(env.num_envs, cfg.gamma, device) if cfg.norm_reward else None
    )

    buf = RolloutBuffer(
        cfg.rollout_len, env.num_envs, env.obs_space, env.action_space, device,
        extra=buffer_extra(cfg, estimator, env),
    )
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)

    batch = cfg.rollout_len * env.num_envs
    n_iters = max(1, cfg.total_steps // batch)
    obs = env.reset(seed=cfg.seed)
    global_step = 0

    for it in range(n_iters):
        cfg._progress = it / n_iters   # needed by surrogates that anneal on progress
        if cfg.anneal_lr:
            frac = 1.0 - cfg._progress
            for g in opt.param_groups:
                g["lr"] = frac * cfg.lr

        obs, steps = collect(env, policy, buf, obs, log, timer, obs_norm, reward_norm)
        global_step += steps

        adv, ret, diag = estimator.compute(
            buf, policy=policy, surrogate=surrogate, cfg=cfg
        )
        if cfg.norm_adv == "batch":
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buf.advantages, buf.returns = adv, ret
        log.add(**diag)

        stop = False
        for epoch in range(cfg.num_epochs):
            estimator.on_epoch_start(
                epoch, buf, policy=policy, surrogate=surrogate, cfg=cfg
            )
            for mb in buf.iter_minibatches(cfg.num_minibatches):
                if cfg.surrogate == "apo" and cfg.apo_resample:
                    cfg._resampled_logratio = _resample_logratio(policy, mb)
                with timer("bwd"):
                    loss, stats = ppo_loss(
                        policy, mb, cfg, surrogate, estimator, value_loss
                    )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                log.add(**{k: v for k, v in stats.items() if not k.startswith("_")})
                log.record("grad/norm", float(gn))
                opt.step()

                if cfg.target_kl is not None and stats["_kl"] > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break

        log.record("diag/explained_variance",
                   explained_variance(stats["_value"], stats["_returns"]))
        log.record("train/lr", opt.param_groups[0]["lr"])
        log.record("perf/sps", steps / max(1e-9, sum(timer.acc.values()) or 1e-9))
        log.add(**timer.drain())
        if it % cfg.log_interval == 0:
            log.dump(global_step)

    log.dump(global_step)
    log.close()
    return policy


def _resample_logratio(policy: Policy, mb) -> torch.Tensor | None:
    """Sample a batch of "unsampled actions" for APO's UARR and return their logratio.

    Actions are resampled from the current policy and then re-evaluated under it. These
    actions **never appeared in the rollout**, which is exactly the anchoring blind spot
    APO constrains. Returns None when the policy has no `dist()`, and the surrogate then
    degrades explicitly.
    """
    if not hasattr(policy, "dist"):
        return None
    with torch.no_grad():
        d = policy.dist(mb.obs)
        a_new = d.sample()
        lp_old = d.log_prob(a_new)
        if lp_old.dim() > 1:
            lp_old = lp_old.sum(-1)
    lp_new, _, _ = policy.evaluate(mb.obs, a_new)
    return lp_new - lp_old


def a2c_config(**kw) -> PPOConfig:
    """A2C is a special case of PPO, not a separate code file."""
    defaults = dict(num_epochs=1, num_minibatches=1, clip_coef=math.inf, gae_lambda=1.0)
    defaults.update(kw)
    return PPOConfig(**defaults)

"""PPO -- flat, linear, readable top to bottom, with no base class.

Every framework-level hyperparameter is listed explicitly in PPOConfig; none are
hard-coded (DESIGN.md §4.6). Hyperparameters belonging to a *swappable component* live on
that component instead, reached through the config spec::

    surrogate: {name: spo, beta: 2.0}

which is the same convention a DIY component uses (see diy/README.md). A2C is just a
config of this algorithm: num_epochs=1, num_minibatches=1, clip_coef=inf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from ..config import Config
from ..logger import Logger
from ..objectives import get_surrogate, get_value_loss
from ..rollout import collect
from ..types import EnvAdapter, Policy
from . import _common
from ._common import buffer_extra  # re-exported: part of the public surface
from .base import algo, alias

__all__ = ["PPOConfig", "ppo_loss", "train", "a2c_config", "buffer_extra"]


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
    # Three accepted forms: a name, {name: ..., kwargs}, or {from: ./my.py:MyClass}.
    # Component-specific hyperparameters go in the spec dict, not in this dataclass --
    # `oprl components` lists what each one accepts.
    surrogate: Any = "ppo"       # policy loss: ppo|tr_ppo|spo|dpo|mdpo|ppo_rpe|apo
    advantage: Any = "gae"       # advantage estimator: gae|dae|vtrace_free_mc
    value_loss: Any = "clipped"  # value loss: clipped|mse|huber
    network: Any = None          # network: {hidden:[...], encoder:..., from:...}
    # Epochs for an estimator's iteration_loss (DAE). 0 = reuse num_epochs.
    num_epochs_critic: int = 0
    buffer: Any = None           # extra buffer fields: {extra: {name: {shape, dtype}}}

    # Written by train() each iteration, for surrogates that anneal (e.g. MDPO).
    _progress: float = 0.0


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


@algo("ppo", PPOConfig, note="PPO (Schulman et al., 2017), pluggable surrogate/advantage")
def train(
    cfg: PPOConfig,
    env: EnvAdapter,
    policy: Policy,
    log: Logger | None = None,
    estimator=None,
) -> Policy:
    # Mechanical setup only -- see algos/_common.py for why it is shared. The training
    # loop itself stays here, in full view.
    rt = _common.setup(cfg, env, policy, log, estimator)
    log, timer, buf, estimator = rt.log, rt.timer, rt.buf, rt.estimator
    surrogate = get_surrogate(cfg.surrogate)
    value_loss = get_value_loss(cfg.value_loss)
    # Optional surrogate hooks, found by attribute check so nothing else implements them.
    # APO needs both: an anchor sampled while theta is still theta_old, then a per-minibatch
    # re-evaluation of it. See objectives/ppo_family.py:APOSurrogate.
    prepare = getattr(surrogate, "prepare", None)
    on_rollout_end = getattr(surrogate, "on_rollout_end", None)

    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    obs = env.reset(seed=cfg.seed)
    global_step = 0
    stats: dict = {}

    for it in range(rt.n_iters):
        _common.begin_iteration(cfg, opt, it, rt.n_iters)

        obs, steps = collect(env, policy, buf, obs, log, timer, rt.obs_norm,
                             rt.reward_norm,
                             extra_writer=getattr(estimator, "write_extra", None))
        global_step += steps

        adv, ret, diag = estimator.compute(
            buf, policy=policy, surrogate=surrogate, cfg=cfg
        )
        if cfg.norm_adv == "batch":
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buf.advantages, buf.returns = adv, ret
        log.add(**diag)

        # Anchor for objectives that reference pi_old beyond the stored logprobs (APO).
        # Placed here, before any parameter update, so `policy` is still the behaviour
        # policy that produced this rollout.
        if on_rollout_end is not None:
            on_rollout_end(policy, buf, cfg)

        # Estimators whose objective needs contiguous time (DAE's telescoping residual)
        # get one optimization pass over the whole rollout, before the minibatch loop
        # shuffles the time axis away.
        it_loss_fn = getattr(estimator, "iteration_loss", None)
        if it_loss_fn is not None:
            for _ in range(cfg.num_epochs_critic or cfg.num_epochs):
                out = it_loss_fn(policy, buf, cfg)
                if out is None:
                    break
                it_loss, it_stats = out
                opt.zero_grad(set_to_none=True)
                it_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt.step()
                log.add(**it_stats)

        stop = False
        for epoch in range(cfg.num_epochs):
            estimator.on_epoch_start(
                epoch, buf, policy=policy, surrogate=surrogate, cfg=cfg
            )
            for mb in buf.iter_minibatches(cfg.num_minibatches):
                if prepare is not None:
                    prepare(policy, mb, cfg)
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

        _common.log_iteration(rt, opt, stats, steps)
        if it % cfg.log_interval == 0:
            log.dump(global_step)

    _common.finish(log, global_step)
    return policy


# A2C is a set of PPO hyperparameters, not a code file (DESIGN.md §4.6). Registering it
# as an alias is what makes `oprl train a2c` work without a second train() function.
A2C_DEFAULTS = dict(num_epochs=1, num_minibatches=1, clip_coef=math.inf, gae_lambda=1.0)

alias("a2c", of="ppo", defaults=A2C_DEFAULTS,
      note="A2C = PPO with one epoch, one minibatch, no clipping, lambda=1")


def a2c_config(**kw) -> PPOConfig:
    """A2C is a special case of PPO, not a separate code file."""
    return PPOConfig(**{**A2C_DEFAULTS, **kw})

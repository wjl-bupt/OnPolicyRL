"""Surrogate protocol plus objectives from published algorithms (DESIGN.md §4.6).

Many recent PPO improvements are neither a hyperparameter change nor a new update
rule -- they **only replace the surrogate objective**. One protocol buys 7 published
algorithms at roughly 15 lines each, and each one can be checked line-by-line against
its paper. That is itself the credibility argument: a reader can verify we got it right.

Doing the same thing in CleanRL would take 7 copy-pasted files.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from ..registry import register


class SurrogateOutput(Protocol):
    """(policy_loss, diagnostics)"""


class Surrogate(Protocol):
    """Given ratio / logp / advantage, return the policy loss.

    Implementations own **only the policy term**. Value loss and the entropy bonus are
    handled once in ppo.py, so no surrogate has to repeat them.
    """

    name: str

    def __call__(
        self,
        ratio: Tensor,
        logp: Tensor,
        logp_old: Tensor,
        adv: Tensor,
        cfg,
    ) -> tuple[Tensor, dict[str, float]]: ...


# --------------------------------------------------------------------------- #
#  PPO (Schulman et al., 2017) -- baseline
# --------------------------------------------------------------------------- #

@register("policy_loss", "ppo")
class ClipSurrogate:
    """The standard PPO clipped surrogate; the control for every variant."""
    name = "ppo"
    def __call__(self, ratio, logp, logp_old, adv, cfg):
        eps = cfg.clip_coef
        loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - eps, 1 + eps)).mean()
        return loss, {"diag/clipfrac": ((ratio - 1).abs() > eps).float().mean().item()}


# --------------------------------------------------------------------------- #
#  TR-PPO / Truly PPO (Wang et al., UAI 2019)
#  https://arxiv.org/abs/1903.07940
# --------------------------------------------------------------------------- #
@register("policy_loss", "tr_ppo")
class TrulyPPOSurrogate:
    """A KL-triggered trust region plus a rollback penalty instead of a hard clip.

    Paper's core idea: when the ratio leaves the trust region, do not zero the gradient
    (what clipping does, which can stall the policy). Instead apply a **negative**
    rollback incentive alpha that pulls the policy back.
    """

    name = "tr_ppo"

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        eps = cfg.clip_coef
        alpha = getattr(cfg, "rollback_alpha", 0.3)
        # Outside the region, -alpha*ratio replaces the clip constant, creating pullback.
        rollback = -alpha * ratio + (1 + alpha) * (1 + eps)
        rollback_neg = -alpha * ratio + (1 + alpha) * (1 - eps)
        out_hi = (ratio > 1 + eps) & (adv > 0)
        out_lo = (ratio < 1 - eps) & (adv < 0)
        eff = torch.where(out_hi, rollback, torch.where(out_lo, rollback_neg, ratio))
        loss = (-adv * eff).mean()
        return loss, {"diag/oob_frac": (out_hi | out_lo).float().mean().item()}


# --------------------------------------------------------------------------- #
#  SPO — Simple Policy Optimization (Xie et al., ICML 2025)
#  https://proceedings.mlr.press/v267/xie25m.html
# --------------------------------------------------------------------------- #


@register("policy_loss", "spo")
class SPOSurrogate:
    """Drop the clip; penalize KL directly in an unconstrained objective which bounds
    the trust region implicitly.

    loss = -adv*ratio + beta * KL(pi_old || pi_new)
    KL uses Schulman's k3 estimator, (r - 1) - log r, which is unbiased and non-negative.
    """

    name = "spo"

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        beta = getattr(cfg, "spo_beta", 1.0)
        logratio = logp - logp_old
        kl = (ratio - 1) - logratio  # k3; non-negative per sample
        loss = (-adv * ratio + beta * kl).mean()
        return loss, {"diag/kl_penalty": kl.mean().item()}


# --------------------------------------------------------------------------- #
#  DPO — Discovered Policy Optimisation (Lu et al., NeurIPS 2022)
#  https://arxiv.org/abs/2210.05639
# --------------------------------------------------------------------------- #


@register("policy_loss", "dpo")
class DPOSurrogate:
    """A closed-form drift objective *discovered* by meta-learning, softened with tanh.

    The paper's drift function has separate branches by advantage sign::

        adv >= 0:  drift = relu((r-1)*A - alpha*tanh((r-1)*A/alpha))
        adv <  0:  drift = relu(log(r)*A - beta*tanh(log(r)*A/beta))

    Objective = -(ratio*A - drift). The tanh saturates the out-of-region penalty,
    avoiding the gradient cliff that clipping creates.
    """

    name = "dpo"

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        a = getattr(cfg, "dpo_alpha", 2.0)
        b = getattr(cfg, "dpo_beta", 0.6)
        logratio = logp - logp_old
        d_pos = torch.relu((ratio - 1) * adv - a * torch.tanh((ratio - 1) * adv / a))
        d_neg = torch.relu(logratio * adv - b * torch.tanh(logratio * adv / b))
        drift = torch.where(adv >= 0, d_pos, d_neg)
        loss = -(ratio * adv - drift).mean()
        return loss, {"diag/drift": drift.mean().item()}


# --------------------------------------------------------------------------- #
#  MDPO — Mirror Descent Policy Optimization (Tomar et al., ICLR 2022)
#  https://openreview.net/forum?id=aBO5SvgSt1
# --------------------------------------------------------------------------- #


@register("policy_loss", "mdpo")
class MDPOSurrogate:
    """Mirror descent: linearized objective plus an explicit reverse-KL proximity term
    whose coefficient is annealed over iterations.

    The differences from SPO are the KL direction (reverse) and the **annealed**
    coefficient. The latter is MDPO's key mechanism: it tightens the trust region as
    training proceeds. Progress is supplied by ppo.py via `cfg._progress`.
    """

    name = "mdpo"

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        t_k = getattr(cfg, "mdpo_tk", 1.0)
        progress = getattr(cfg, "_progress", 0.0)  # 0 -> 1
        beta = t_k * (1.0 - progress)  # the paper's 1 - k/K schedule
        # Single-sample estimate of reverse KL(pi_new || pi_old).
        logratio = logp - logp_old
        rev_kl = ratio * logratio - (ratio - 1)
        loss = (-adv * ratio + rev_kl / max(beta, 1e-8)).mean()
        return loss, {"diag/rev_kl": rev_kl.mean().item(), "diag/mdpo_beta": beta}


# --------------------------------------------------------------------------- #
#  PPO-RPE — Relative Pearson Divergence (Kobayashi, ICRA 2021)
#  https://arxiv.org/abs/2010.03290
# --------------------------------------------------------------------------- #


@register("policy_loss", "ppo_rpe")
class RPESurrogate:
    """Regularize with the relative Pearson divergence, giving a clip target that is
    symmetric in the density ratio.

    That symmetry constrains positive and negative advantages equally, whereas the
    standard clip is asymmetric.
    """

    name = "ppo_rpe"

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        eps = cfg.clip_coef
        alpha = getattr(cfg, "rpe_alpha", 0.5)
        # Relative density ratio: r / (alpha*r + (1-alpha)).
        rel = ratio / (alpha * ratio + (1 - alpha))
        rel_clip = rel.clamp(1 - eps, 1 + eps)
        loss = torch.max(-adv * rel, -adv * rel_clip).mean()
        return loss, {"diag/rel_ratio": rel.mean().item()}


# --------------------------------------------------------------------------- #
#  LPO / Mirror Learning (Kuba et al., ICML 2022) -- **not implemented, see below**
#  https://proceedings.mlr.press/v162/grudzien22a.html
# --------------------------------------------------------------------------- #
#
# Mirror Learning is a **theoretical framework**: it proves monotonic improvement for
# any drift plus neighbourhood operator satisfying its conditions, with PPO as one
# instance. Its LPO instance uses a **meta-learned neural network drift**, not a closed
# form -- so it is not a 15-line surrogate and does not belong in this file.
#
# DPO (Discovered Policy Optimisation, NeurIPS 2022) is the **closed-form distillation**
# of that meta-learning result and is implemented above. For Mirror Learning's practical
# benefit, use DPO.
#
# An invented "logistic drift" once sat here. It ran fine but did not learn (CartPole
# return 24.4 after 60k steps, random is about 20), so it was removed.
# Lesson: the surrogate protocol makes adding algorithms cheap, which makes the
# discipline of implementing the actual published formula more important, not less.
# Every implementation must be checkable against its paper; anything that is not does
# not belong here.

# --------------------------------------------------------------------------- #
#  APO — Anchored Policy Optimization (Luo et al., Neural Networks 205, 2026)
#  https://doi.org/10.1016/j.neunet.2026.109476
# --------------------------------------------------------------------------- #


@register("policy_loss", "apo")
class APOSurrogate:
    """PPO clip plus UARR: constrain ratio ~ 1 on **unsampled actions**, fixing the
    "anchoring blindness" the paper identifies.

    Motivation: standard PPO only constrains the ratio of actions that were sampled, so
    unsampled actions may drift freely. In continuous action spaces this is a real
    problem (directly relevant to the MuJoCo / Isaac paths).

    UARR needs the logratio of resampled actions, supplied by ppo.py via
    `cfg._resampled_logratio`. When absent this degrades to standard PPO -- explicitly,
    not silently.
    """

    name = "apo"

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        eps = cfg.clip_coef
        base = torch.max(-adv * ratio, -adv * ratio.clamp(1 - eps, 1 + eps)).mean()
        resampled = getattr(cfg, "_resampled_logratio", None)
        if resampled is None:
            return base, {"diag/uarr": 0.0, "diag/apo_degraded": 1.0}
        # UARR: the ratio on unsampled actions should stay near 1.
        uarr = ((resampled.exp() - 1.0) ** 2).mean()
        lam = getattr(cfg, "apo_uarr_coef", 0.1)
        return base + lam * uarr, {"diag/uarr": uarr.item(), "diag/apo_degraded": 0.0}


# --------------------------------------------------------------------------- #

SURROGATES: dict[str, Surrogate] = {
    s.name: s()  # type: ignore[abstract]
    for s in (
        ClipSurrogate,
        TrulyPPOSurrogate,
        SPOSurrogate,
        DPOSurrogate,
        MDPOSurrogate,
        RPESurrogate,
        APOSurrogate,
    )
}


def get_surrogate(spec) -> Surrogate:
    """Accepts a name, a dict (with kwargs or `from`), or an object."""
    from ..registry import build

    if not isinstance(spec, (str, dict)):
        return spec
    return build("policy_loss", spec)

"""Surrogate protocol plus objectives from published algorithms (DESIGN.md §4.6).

Many recent PPO improvements are neither a hyperparameter change nor a new update
rule -- they **only replace the surrogate objective**. One protocol buys 7 published
algorithms at roughly 15 lines each, and each one can be checked line-by-line against
its paper. That is itself the credibility argument: a reader can verify we got it right.

Doing the same thing in CleanRL would take 7 copy-pasted files.

**Where a surrogate's hyperparameters live.** On the surrogate, taken through
`__init__`, and supplied from the config spec::

    surrogate: {name: dpo, alpha: 2.0, beta: 0.6}

They used to be fields on `PPOConfig`, which meant every new algorithm widened a
dataclass shared by all of them, and built-in components followed a different convention
from DIY ones (diy/README.md mandates `__init__`). `cfg` is still passed to `__call__`
because objectives legitimately read *framework* settings from it -- `clip_coef`,
`_progress` -- but nothing algorithm-private is read from it any more.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from ..registry import register


class Surrogate(Protocol):
    """Given ratio / logp / advantage, return the policy loss.

    Implementations own **only the policy term**. Value loss and the entropy bonus are
    handled once in ppo.py, so no surrogate has to repeat them.

    An implementation may optionally define `prepare(policy, mb, cfg)`, called by the
    algorithm once per minibatch before the loss. APO uses it; nothing else needs it, and
    absence is detected by attribute check rather than by a name comparison.
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

    def __init__(self, rollback_alpha: float = 0.3):
        self.rollback_alpha = float(rollback_alpha)

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        eps = cfg.clip_coef
        alpha = self.rollback_alpha
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
    """Drop the clip; apply an advantage-weighted quadratic penalty on the ratio, which
    bounds the trust region implicitly::

        loss = -mean( A*r  -  beta * |A| * (r - 1)^2 / (2*eps) )

    Every term sits inside **one** mean, because the penalty is weighted per sample by |A|:
    a transition with a large advantage is held closer to r = 1 than one with a small
    advantage. Reducing the two terms separately would discard that weighting (and, if the
    per-sample factor were multiplied onto an already-reduced mean, would return a vector
    instead of a scalar loss).

    Note `(r-1)^2 / 2` is the second-order expansion of Schulman's k3 KL estimator, so this
    is an advantage-weighted KL penalty with coefficient 1/eps rather than a uniform one.

    `eps` is floored at 1e-3, as in the reference implementation: the penalty coefficient
    diverges as `clip_coef` shrinks, and without the floor a small clip_coef silently
    becomes an enormous penalty.
    """

    name = "spo"

    def __init__(self, beta: float = 1.0):
        """beta scales the penalty. 1.0 is the published form; larger tightens the region."""
        self.beta = float(beta)

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        eps = max(float(cfg.clip_coef), 1e-3)
        penalty = adv.abs() * (ratio - 1.0) ** 2 / (2.0 * eps)
        loss = -(adv * ratio - self.beta * penalty).mean()
        return loss, {
            "diag/spo_penalty": penalty.mean().item(),
            "diag/ratio_drift": (ratio - 1.0).abs().mean().item(),
        }


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

    def __init__(self, alpha: float = 2.0, beta: float = 0.6):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        a, b = self.alpha, self.beta
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
    training proceeds. Progress is a framework quantity, supplied by ppo.py via
    `cfg._progress`.
    """

    name = "mdpo"

    def __init__(self, t_k: float = 1.0):
        self.t_k = float(t_k)

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        progress = getattr(cfg, "_progress", 0.0)  # 0 -> 1
        beta = self.t_k * (1.0 - progress)  # the paper's 1 - k/K schedule
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

    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        eps = cfg.clip_coef
        alpha = self.alpha
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

    UARR needs actions that never appeared in the rollout, evaluated against the **rollout
    policy** pi_old. So this surrogate uses two optional hooks the algorithm calls for it:

      `on_rollout_end(policy, buf, cfg)`  once per iteration, while theta is still
          theta_old -- samples the anchor actions and records log pi_old for them.
      `prepare(policy, mb, cfg)`  once per minibatch -- re-evaluates those same actions
          under the current theta, giving a logratio that is actually non-zero.

    Both are found by attribute check, so no other surrogate implements them.

    The penalty weight is either uniform (`uarr_coef=0.1`) or, with `uarr_coef="auto"`,
    proportional to pi_old of the anchor action -- see `__init__`.

    Two bugs previously lived here. First, ppo.py dispatched on `cfg.surrogate == "apo"`,
    so writing the surrogate in dict form (`{name: apo}`) skipped UARR silently. Second,
    and worse, the resampling drew actions from the *current* policy and scored them with
    that same policy, so logratio was identically 0 and UARR contributed no gradient at
    all -- the term was decorative. Anchoring on pi_old is what makes it real.
    """

    name = "apo"

    # `uarr_coef="auto"` selects the per-sample weighting described in `__init__`. It is a
    # separate sentinel rather than an out-of-range number so that `uarr_coef=0.0` keeps its
    # obvious meaning of "penalty off".
    AUTO = "auto"

    def __init__(self, uarr_coef: float | str = 0.1, resample: bool = True):
        """
        Args:
            uarr_coef: weight on the UARR penalty.
                A number applies it uniformly (0.0 turns the penalty off).
                `"auto"` weights each sample by **0.5 * pi_old(anchor action | s)**, so
                anchoring is enforced hardest in states where the rollout policy already put
                real probability mass on the action being constrained, and barely at all in
                states where it did not. Note this is the probability of the *anchor* action,
                not of the action that was actually taken -- the anchor is what UARR
                constrains, so it is what should set the weight.
            resample: False disables anchoring entirely; the objective is then plain PPO and
                reports `diag/apo_degraded == 1`.
        """
        self.auto_coef = isinstance(uarr_coef, str) and uarr_coef.lower() == self.AUTO
        if self.auto_coef:
            self.uarr_coef: float = float("nan")   # unused on the auto path
        else:
            self.uarr_coef = float(uarr_coef)
            if self.uarr_coef < 0.0:
                raise ValueError(
                    f"uarr_coef must be >= 0 or {self.AUTO!r}, got {uarr_coef!r}"
                )
        self.resample = bool(resample)
        # (anchor actions, log pi_old of them) for the whole rollout, set once per iteration.
        self._anchor: tuple[Tensor, Tensor] | None = None
        # Per-minibatch: the anchor's logratio, and log pi_old sliced to match it.
        self._logratio: Tensor | None = None
        self._anchor_logp_old: Tensor | None = None

    def on_rollout_end(self, policy, buf, cfg) -> None:
        """Sample the anchor actions under pi_old and cache their log-probabilities.

        Called before the epoch loop, so `policy` is still the behaviour policy. These
        actions are deliberately *not* the ones in the buffer -- they are the unsampled
        region APO constrains.
        """
        self._anchor = None
        if not self.resample or not hasattr(policy, "dist"):
            return
        from ..tree import tree_flatten_time

        with torch.no_grad():
            d = policy.dist(tree_flatten_time(buf.obs))
            a = d.sample()
            lp = d.log_prob(a)
            if lp.dim() > 1:
                lp = lp.sum(-1)
        self._anchor = (a, lp)

    def prepare(self, policy, mb, cfg) -> None:
        """Re-evaluate the anchor actions under the current policy for this minibatch."""
        self._logratio = None
        self._anchor_logp_old = None
        if self._anchor is None:
            return
        actions, lp_old = self._anchor
        idx = mb.get("_flat_idx")
        if idx is not None:
            actions, lp_old = actions[idx], lp_old[idx]
        lp_new, _, _ = policy.evaluate(mb.obs, actions)
        self._logratio = lp_new - lp_old
        self._anchor_logp_old = lp_old

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        eps = cfg.clip_coef
        base = torch.max(-adv * ratio, -adv * ratio.clamp(1 - eps, 1 + eps)).mean()
        resampled = self._logratio
        if resampled is None or (not self.auto_coef and self.uarr_coef == 0.0):
            return base, {"diag/uarr": 0.0, "diag/apo_degraded": 1.0}

        # UARR: the ratio on unsampled actions should stay near 1.
        sq = (resampled.exp() - 1.0) ** 2
        if self.auto_coef:
            # Per-sample weight from pi_old of the **anchor** action, detached: it is a
            # weighting, not a quantity to backpropagate through.
            coef = 0.5 * self._anchor_logp_old.exp().detach()
            uarr = (sq * coef).mean()
            coef_mean = float(coef.mean())
        else:
            uarr = sq.mean() * self.uarr_coef
            coef_mean = self.uarr_coef

        return base + uarr, {
            "diag/uarr": uarr.item(),
            "diag/uarr_coef": coef_mean,
            # A flag, not a magnitude: 0 = UARR active, 1 = degraded to plain PPO.
            "diag/apo_degraded": 0.0,
        }


# --------------------------------------------------------------------------- #
#  PPO + explicit KL penalty (Schulman et al., 2017, the paper's penalty variant)
#  https://arxiv.org/abs/1707.06347  (eq. 8, "adaptive KL penalty" without adaptation)
# --------------------------------------------------------------------------- #


@register("policy_loss", "ppo_kl")
class ClipPlusKLSurrogate:
    """PPO's clipped surrogate **plus** an explicit KL penalty -- clip and penalty, not
    penalty instead of clip.

    The distinction from `spo` matters and is easy to miss: SPO **removes** the clip and
    relies on the KL term alone, whereas this keeps both. They are not interchangeable, and
    a lambda-selection ablation run against one cannot be compared with the other.

    Exists because the GA2E reference implementation had a `kl_coef` code path
    (`_train_with_kl_loss`) that was exactly this objective; without it, results from those
    runs would have no counterpart here. KL uses Schulman's k3 estimator, so it is
    differentiable and non-negative per sample.
    """

    name = "ppo_kl"

    def __init__(self, kl_coef: float = 0.01):
        self.kl_coef = float(kl_coef)

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        eps = cfg.clip_coef
        clipped = torch.max(-adv * ratio, -adv * ratio.clamp(1 - eps, 1 + eps)).mean()
        logratio = logp - logp_old
        kl = ((ratio - 1) - logratio).mean()   # k3
        return clipped + self.kl_coef * kl, {
            "diag/clipfrac": ((ratio - 1).abs() > eps).float().mean().item(),
            "diag/kl_penalty": kl.item(),
        }


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
        ClipPlusKLSurrogate,
    )
}


def get_surrogate(spec) -> Surrogate:
    """Accepts a name, a dict (with kwargs or `from`), or an object.

    A bare name builds a **fresh instance** rather than returning the shared one in
    SURROGATES, because a surrogate may hold per-minibatch state (APO's cached logratio).
    Sharing one instance across concurrent runs in a single process would let them
    interfere.
    """
    from ..registry import build

    if not isinstance(spec, (str, dict)):
        return spec
    return build("policy_loss", spec)

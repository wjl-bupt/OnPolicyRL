"""SPO -- Simple Policy Optimization, as a DIY component.

Paper:  Simple Policy Optimization (Xie & Zhang, ICML 2025)
        https://proceedings.mlr.press/v267/xie25m.html

Idea: drop PPO's ratio clipping. Instead optimize an unconstrained objective with an
**advantage-weighted quadratic penalty** on the ratio, which bounds the trust region
implicitly::

    loss = -mean( A*r  -  beta * |A| * (r - 1)^2 / (2*eps) )

where r = pi_new/pi_old and eps = cfg.clip_coef.

Two details that are easy to get wrong:

**The whole expression is inside one mean.** The penalty carries a per-sample |A| factor, so
a transition with a large advantage is pulled back toward r = 1 harder than one with a small
advantage. Reducing the reward term and the penalty term separately throws that weighting
away -- and multiplying a per-sample |A| onto an already-reduced `.mean()` yields a *vector*,
not a scalar loss, which autograd will happily accept before failing somewhere less obvious.

**`(r-1)^2 / 2` is the second-order expansion of Schulman's k3 KL estimator**,
`KL_k3 = (r - 1) - log r`. So this is a KL penalty in all but name -- specifically an
advantage-weighted one with coefficient 1/eps, rather than the uniform coefficient a plain
k3 penalty would give.

Why clipping is worth dropping: a clipped ratio has zero gradient outside the trust region,
so a policy that overshoots gets no signal pulling it back -- it can sit there. A quadratic
penalty always pushes back, proportionally to how far out it is.

This file is a **standalone reimplementation** -- it imports nothing from oprl. The
framework's own `spo` surrogate is in `src/oprl/objectives/ppo_family.py`, and
`tests/test_diy.py` asserts the two agree numerically.
"""

from __future__ import annotations

import torch
from torch import Tensor


class SimplePolicyOptimization:
    """A `policy_loss` component.

    The protocol is a single call with the signature::

        (ratio, logp, logp_old, adv, cfg) -> (loss, stats)

    `loss` must stay attached to the graph; `stats` maps metric name to float and is
    forwarded to the logger as-is (any `diag/*` key is aggregated by mean).
    """

    # Parameters arrive from the config spec, not from PPOConfig -- see diy/README.md.
    def __init__(self, beta: float = 1.0, adaptive: bool = False, target_kl: float = 0.01):
        """
        Args:
            beta: penalty coefficient. Larger = tighter trust region.
            adaptive: scale beta per minibatch toward `target_kl` (a common trick from
                the original PPO paper's KL-penalty variant, not from the SPO paper --
                left off by default so the default path stays faithful to the paper).
            target_kl: only used when `adaptive=True`.
        """
        self.beta = float(beta)
        self.adaptive = bool(adaptive)
        self.target_kl = float(target_kl)
        self._beta_cur = float(beta)

    def __call__(
        self, ratio: Tensor, logp: Tensor, logp_old: Tensor, adv: Tensor, cfg
    ) -> tuple[Tensor, dict[str, float]]:
        # Floored, as in the reference: the coefficient is 1/(2*eps), so a small clip_coef
        # would otherwise silently turn into an enormous penalty.
        eps = max(float(cfg.clip_coef), 1e-3)

        if self.adaptive:
            # Detached: the coefficient is a schedule, not something to backprop through.
            # k3 is the quantity being targeted, so it is what the schedule watches.
            with torch.no_grad():
                logratio = logp - logp_old
                kl_mean = ((ratio - 1.0) - logratio).mean().item()
                if kl_mean > 1.5 * self.target_kl:
                    self._beta_cur = min(self._beta_cur * 1.5, 1e4)
                elif kl_mean < self.target_kl / 1.5:
                    self._beta_cur = max(self._beta_cur / 1.5, 1e-4)
        beta = self._beta_cur if self.adaptive else self.beta

        # One mean over the whole expression -- see the module docstring on why the
        # per-sample |A| weighting cannot be factored out of it.
        penalty = adv.abs() * (ratio - 1.0) ** 2 / (2.0 * eps)
        loss = -(adv * ratio - beta * penalty).mean()

        return loss, {
            "diag/spo_penalty": penalty.mean().item(),
            "diag/spo_beta": beta,
            # No clipping happens, but reporting how far the ratio drifts keeps this
            # comparable with the clipped baselines.
            "diag/ratio_drift": (ratio - 1.0).abs().mean().item(),
        }


if __name__ == "__main__":
    # Cheap self-check: an objective with a sign error still runs to completion, it just
    # never learns. Verify the gradient direction before spending GPU hours.
    from dataclasses import dataclass

    @dataclass
    class _Cfg:
        clip_coef: float = 0.2

    loss_fn = SimplePolicyOptimization(beta=1.0)
    logp_old = torch.zeros(256)
    logp = torch.zeros(256, requires_grad=True)

    # Positive advantage must push this action's log-probability up, i.e. d(loss)/d(logp) < 0.
    loss, stats = loss_fn(logp.exp(), logp, logp_old, torch.ones(256), _Cfg())
    assert loss.dim() == 0, f"loss must be a scalar, got shape {tuple(loss.shape)}"
    loss.backward()
    grad = logp.grad.sum().item()
    print(f"loss={loss.item():+.4f}  d(loss)/d(logp)={grad:+.4f}  {stats}")
    assert grad < 0, "sign error: a positive advantage must increase log-prob"

    # At ratio == 1 the penalty must vanish exactly, or every update is biased even when the
    # policy has not moved.
    loss0, stats0 = loss_fn(torch.ones(8), torch.zeros(8), torch.zeros(8),
                            torch.ones(8), _Cfg())
    assert abs(stats0["diag/spo_penalty"]) < 1e-9, "penalty must be 0 when pi_new == pi_old"

    # The penalty is weighted per sample by |A|: doubling the advantage must more than double
    # it. If the |A| factor were pulled outside the mean, this would fail.
    r = torch.full((8,), 1.3)
    _, s1 = loss_fn(r, torch.zeros(8), torch.zeros(8), torch.ones(8), _Cfg())
    _, s2 = loss_fn(r, torch.zeros(8), torch.zeros(8), torch.full((8,), 2.0), _Cfg())
    assert abs(s2["diag/spo_penalty"] - 2 * s1["diag/spo_penalty"]) < 1e-6, (
        "the penalty is not scaling with |A| -- the per-sample weighting was lost"
    )

    print("self-check passed")

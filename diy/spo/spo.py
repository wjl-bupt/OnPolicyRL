"""SPO -- Simple Policy Optimization, as a DIY component.

Paper:  Simple Policy Optimization (Xie & Zhang, ICML 2025)
        https://proceedings.mlr.press/v267/xie25m.html

Idea: drop PPO's ratio clipping. Instead optimize an unconstrained objective that
penalizes KL(pi_old || pi_new) directly, which bounds the trust region implicitly::

    loss = -A * r  +  beta * KL(pi_old || pi_new)

where r = pi_new/pi_old. The KL uses Schulman's k3 estimator::

    KL_k3 = (r - 1) - log r

k3 is unbiased and **non-negative for every sample**, unlike the naive `-log r` estimator
which is only non-negative in expectation and can therefore hand back a negative penalty
on individual minibatches.

Why clipping is worth dropping: a clipped ratio has zero gradient outside the trust
region, so a policy that overshoots gets no signal pulling it back -- it can sit there.
A KL penalty always pushes back, proportionally to how far out it is.

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
            beta: KL penalty coefficient. Larger = tighter trust region.
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
        logratio = logp - logp_old

        # k3 estimator: non-negative per sample, so the penalty never flips sign.
        kl = (ratio - 1.0) - logratio

        if self.adaptive:
            # Detached: the coefficient is a schedule, not something to backprop through.
            with torch.no_grad():
                kl_mean = kl.mean().item()
                if kl_mean > 1.5 * self.target_kl:
                    self._beta_cur = min(self._beta_cur * 1.5, 1e4)
                elif kl_mean < self.target_kl / 1.5:
                    self._beta_cur = max(self._beta_cur / 1.5, 1e-4)
        beta = self._beta_cur if self.adaptive else self.beta

        loss = (-adv * ratio + beta * kl).mean()

        return loss, {
            "diag/kl_penalty": kl.mean().item(),
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
    loss.backward()
    grad = logp.grad.sum().item()
    print(f"loss={loss.item():+.4f}  d(loss)/d(logp)={grad:+.4f}  {stats}")
    assert grad < 0, "sign error: a positive advantage must increase log-prob"

    # At ratio == 1 the KL penalty must vanish exactly.
    loss0, stats0 = loss_fn(torch.ones(8), torch.zeros(8), torch.zeros(8),
                            torch.zeros(8), _Cfg())
    assert abs(stats0["diag/kl_penalty"]) < 1e-9, "KL must be 0 when pi_new == pi_old"

    print("self-check passed")

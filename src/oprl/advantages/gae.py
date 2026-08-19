"""Advantage estimation. GAE is the default implementation (DESIGN.md §4.1 / §4.7).

**This module carries the framework's most important correctness property.** The two
masks do different jobs:
  - the value bootstrap is cut only by `terminated` (a truncated state still has value)
  - the advantage recursion is cut by `terminated | truncated`

Collapsing them into a single `done` is the most widespread on-policy bug in the field.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..types import Masks
from .base import BaseEstimator, register


def gae(
    rewards: Tensor,
    values: Tensor,
    bootstrap_value: Tensor,
    masks: Masks,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Generalized Advantage Estimation.

    Args:
        rewards: [T, N]
        values: [T, N] -- V(s_t)
        bootstrap_value: [N] -- V(s_T)
        masks: all three masks are required; this is a deliberate type constraint
        gamma, lam: discount and GAE coefficient

    Returns:
        (advantages, returns), both [T, N]
    """
    T, N = rewards.shape
    if values.shape != (T, N):
        raise ValueError(f"expected values of shape {(T, N)}, got {tuple(values.shape)}")

    term = masks.terminated.to(rewards.dtype)
    trunc = masks.truncated.to(rewards.dtype)

    # Build V(s_{t+1}); the final step uses bootstrap_value.
    next_values = torch.cat([values[1:], bootstrap_value.reshape(1, N)], dim=0)

    # Bootstrap mask: cut only by `terminated`.
    #   When truncated, (1 - term) == 1 and gamma * V(s_next) is kept -- the key point.
    delta = rewards + gamma * next_values * (1.0 - term) - values

    # Recursion mask: either flag ends the trajectory.
    cont = 1.0 - torch.clamp(term + trunc, max=1.0)

    advantages = torch.zeros_like(rewards)
    last = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)
    for t in range(T - 1, -1, -1):
        last = delta[t] + gamma * lam * cont[t] * last
        advantages[t] = last

    returns = advantages + values
    return advantages, returns


@register("gae")
class GAE(BaseEstimator):
    """Default AdvantageEstimator -- a stateless pure function."""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95, cfg=None):
        if cfg is not None:
            gamma = getattr(cfg, "gamma", gamma)
            lam = getattr(cfg, "gae_lambda", lam)
        self.gamma = gamma
        self.lam = lam

    def compute(self, buf, policy=None, surrogate=None, cfg=None):
        adv, ret = gae(
            buf["reward"][: buf.T],
            buf.values,
            buf.bootstrap_value,
            buf.masks,
            self.gamma,
            self.lam,
        )
        return adv, ret, {}


@register("vtrace_free_mc")
class MonteCarloAdvantage(BaseEstimator):
    """GAE with lam=1, i.e. Monte-Carlo return minus baseline. A lam-boundary control.

    Note this is **not** off-policy V-trace: there is no importance-sampling
    correction, so the "data always comes from the current policy" invariant holds
    (DESIGN.md §2).
    """

    def __init__(self, gamma: float = 0.99, cfg=None):
        self.gamma = getattr(cfg, "gamma", gamma) if cfg is not None else gamma

    def compute(self, buf, policy=None, surrogate=None, cfg=None):
        adv, ret = gae(
            buf["reward"][: buf.T], buf.values, buf.bootstrap_value,
            buf.masks, self.gamma, 1.0,
        )
        return adv, ret, {}

"""Value losses -- pluggable via `value_loss: clipped | mse | huber | {from: ...}`.

The signature is uniformly (value, returns, value_old, cfg) -> (loss, stats).
A custom implementation only has to match that signature; it inherits nothing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..registry import register


@register("value_loss", "clipped")
class ClippedValueLoss:
    """The original PPO paper's choice: clip the value target by `clip_coef` too."""

    def __call__(self, value: Tensor, returns: Tensor, value_old: Tensor, cfg):
        eps = getattr(cfg, "clip_coef", 0.2)
        v_clipped = value_old + (value - value_old).clamp(-eps, eps)
        loss = 0.5 * torch.max(
            (value - returns) ** 2, (v_clipped - returns) ** 2
        ).mean()
        return loss, {}


@register("value_loss", "mse")
class MSEValueLoss:
    def __call__(self, value: Tensor, returns: Tensor, value_old: Tensor, cfg):
        return 0.5 * ((value - returns) ** 2).mean(), {}


@register("value_loss", "huber")
class HuberValueLoss:
    """More robust to outlying value targets; worth trying on large-reward tasks."""

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(self, value: Tensor, returns: Tensor, value_old: Tensor, cfg):
        loss = F.huber_loss(value, returns, delta=self.delta)
        return loss, {"diag/value_huber_delta": self.delta}

# -*- encoding: utf-8 -*-
"""Weight initialization for the network axis (S0).

Orthogonal init with per-role gains is the SB3/CleanRL convention and is
load-bearing for PPO -- not a cosmetic default:

* hidden layers at gain sqrt(2) preserve activation variance through tanh/relu
  stacks (the standard choice for both);
* the policy head at gain 0.01 starts near-uniform (tiny logits / tiny mu), so
  the first updates are not dominated by an arbitrary initial action preference;
* the value head at gain 1.0 starts as an unbiased small-variance estimator.

A PPO that "half-learns" and plateaus is very often a policy head left at the
default init. gains are chosen by the assembler (actor_critic.build_policy), not
hard-coded per layer here.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn


def layer_init(layer: nn.Module, gain: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    """Orthogonal-init a Linear/Conv layer in place and return it (so it reads
    inline: ``layer_init(nn.Linear(i, o), gain=0.01)``). Bias-free layers (a
    Conv with bias=False) skip the bias fill."""
    nn.init.orthogonal_(layer.weight, gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias)
    return layer

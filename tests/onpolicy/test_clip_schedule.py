"""Clip-range schedule -- onpolicy/ppo.py ClipPolicyLoss (S9).

The clip range can decay over training: `constant` holds epsilon; `linear` anneals
epsilon -> epsilon_min as batch.progress runs 0 -> 1, floored so it never reaches 0
(a 0 clip range removes PPO's trust region and collapses it to vanilla PG).
"""

import pytest
import torch as th

from onpolicy.ppo import Batch, ClipPolicyLoss, PPOConfig


def _batch(progress):
    # a minimal Batch: the surrogate only reads advantages / logp / old_log_prob
    # (and progress for the schedule); the rest are placeholders.
    n = 4
    return Batch(
        observations=th.zeros(n, 2), actions=th.zeros(n, dtype=th.long),
        old_log_prob=th.zeros(n), old_values=th.zeros(n),
        advantages=th.ones(n), returns=th.zeros(n),
        logp=th.zeros(n), entropy=th.zeros(n), value=th.zeros(n),
        progress=progress,
    )


def test_constant_schedule_holds_epsilon():
    loss = ClipPolicyLoss(0.2, schedule="constant")
    assert loss._current_eps(0.0) == 0.2
    assert loss._current_eps(1.0) == 0.2          # progress has no effect


def test_linear_schedule_decays_to_min_not_zero():
    loss = ClipPolicyLoss(0.2, schedule="linear", epsilon_min=1e-3)
    assert loss._current_eps(0.0) == 0.2                   # start = epsilon
    assert abs(loss._current_eps(1.0) - 1e-3) < 1e-12      # end = epsilon_min...
    assert loss._current_eps(1.0) > 0.0                    # ...never 0
    assert 1e-3 < loss._current_eps(0.5) < 0.2            # monotone in between


def test_clip_range_reported_in_stats():
    loss = ClipPolicyLoss(0.2, schedule="linear", epsilon_min=1e-3)
    _loss, stats = loss(_batch(progress=1.0))
    assert abs(stats["train/clip_range"] - 1e-3) < 1e-12   # the eps actually used
    assert "train/clipfrac" in stats


def test_config_rejects_bad_schedule():
    with pytest.raises(ValueError, match="clip_schedule"):
        PPOConfig(clip_schedule="cosine")

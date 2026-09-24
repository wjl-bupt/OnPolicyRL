"""Deterministic evaluation -- common/evaluate.py (L1).

evaluate() runs a policy on a FRESH env and reports raw-return `eval/*` metrics.
Determinism (default) means two same-seed calls score identically, and eval never
disturbs the training RNG stream.
"""

import numpy as np
import pytest

pytest.importorskip("gymnasium")
import gymnasium as gym

from architecture.actor_critic import build_policy
from common.env.adapter import GymVectorAdapter
from common.env.make_env import make_env
from common.evaluate import evaluate


def _spaces():
    return (gym.spaces.Box(-1, 1, (4,), dtype=np.float32), gym.spaces.Discrete(2))


def _factory():
    return lambda: GymVectorAdapter(make_env("CartPole-v1", num_envs=1, seed=0))


def test_evaluate_reports_metrics():
    pol = build_policy(*_spaces())
    m = evaluate(pol, _factory(), episodes=3, seed=0)
    for k in ("eval/ep_rew_mean", "eval/ep_rew_std", "eval/ep_len_mean",
              "eval/n_episodes"):
        assert k in m
    assert m["eval/n_episodes"] == 3
    assert np.isfinite(m["eval/ep_rew_mean"])
    assert np.isfinite(m["eval/ep_len_mean"])


def test_deterministic_eval_is_reproducible():
    pol = build_policy(*_spaces())           # one policy, fixed weights
    a = evaluate(pol, _factory(), episodes=3, seed=0, deterministic=True)
    b = evaluate(pol, _factory(), episodes=3, seed=0, deterministic=True)
    assert a["eval/ep_rew_mean"] == b["eval/ep_rew_mean"]
    assert a["eval/ep_len_mean"] == b["eval/ep_len_mean"]

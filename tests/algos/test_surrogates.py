"""Surrogate and estimator protocol tests.

Key assertion: **every surrogate must actually learn**, not merely run to completion.
An objective with a numerical mistake still finishes fine -- it just does not learn.
"""

import pytest
import torch

import oprl
from oprl.advantages import ESTIMATORS, get_estimator
from oprl.algos import ppo
from oprl.objectives import SURROGATES, get_surrogate

ALL_SURROGATES = sorted(SURROGATES)


def test_registry_contents():
    """The expected surrogates and estimators are registered."""
    assert set(ALL_SURROGATES) == {
        "ppo", "tr_ppo", "spo", "dpo", "mdpo", "ppo_rpe", "apo"
    }
    assert "gae" in ESTIMATORS


def test_get_accepts_name_or_object():
    """A string is only shorthand; passing the object is always equivalent."""
    s = get_surrogate("dpo")
    assert get_surrogate(s) is s
    with pytest.raises(KeyError):
        get_surrogate("nonexistent")
    with pytest.raises(KeyError):
        get_estimator("nonexistent")


@pytest.mark.parametrize("name", ALL_SURROGATES)
def test_surrogate_gradient_direction(name):
    """Basic correctness: a positive advantage must push the action's probability up.

    That is d(loss)/d(logp) < 0, so a sign error in any surrogate is caught here.
    """
    cfg = ppo.PPOConfig(clip_coef=0.2)
    logp_old = torch.zeros(64)
    logp = torch.zeros(64, requires_grad=True)
    adv = torch.ones(64)  # all-positive advantage
    loss, stats = get_surrogate(name)(logp.exp() / 1.0, logp, logp_old, adv, cfg)
    loss.backward()
    assert logp.grad is not None
    assert logp.grad.sum() < 0, f"{name}: wrong gradient direction for advantage > 0"
    assert isinstance(stats, dict)


@pytest.mark.parametrize("name", ALL_SURROGATES)
def test_surrogate_runs(name):
    """Every surrogate runs the full training loop."""
    env = oprl.make_env("CartPole-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    cfg = ppo.PPOConfig(total_steps=4 * 32 * 2, num_envs=4, rollout_len=32,
                        device="cpu", num_epochs=2, surrogate=name)
    ppo.train(cfg, env, policy, oprl.Logger(sinks=[]))
    env.close()


def test_estimator_swappable():
    """Advantage estimators are pluggable via registry dispatch."""
    env = oprl.make_env("CartPole-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    cfg = ppo.PPOConfig(total_steps=4 * 32 * 2, num_envs=4, rollout_len=32,
                        device="cpu", num_epochs=2, advantage="vtrace_free_mc")
    ppo.train(cfg, env, policy, oprl.Logger(sinks=[]))
    env.close()


def test_surrogate_and_estimator_compose():
    """The two protocols are orthogonal and compose freely -- the ablation space."""
    env = oprl.make_env("CartPole-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    cfg = ppo.PPOConfig(total_steps=4 * 32 * 2, num_envs=4, rollout_len=32,
                        device="cpu", num_epochs=2,
                        surrogate="dpo", advantage="vtrace_free_mc")
    ppo.train(cfg, env, policy, oprl.Logger(sinks=[]))
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize("name", ALL_SURROGATES)
def test_surrogate_learns(name):
    """**Core test**: each surrogate must clearly beat random (~20) on CartPole.

    An objective with a numerical error runs without complaint; only this test catches it.
    """
    env = oprl.make_env("CartPole-v1", num_envs=8, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    cfg = ppo.PPOConfig(total_steps=60_000, num_envs=8, rollout_len=128,
                        device="cpu", num_epochs=4, num_minibatches=4,
                        seed=1, surrogate=name)
    log = oprl.Logger(sinks=[])
    ppo.train(cfg, env, policy, log)
    env.close()
    ret = sum(log.ep_returns) / len(log.ep_returns)
    assert ret > 60, f"surrogate '{name}' failed to learn CartPole: mean return {ret:.1f}"

"""Learning-capability tests: an algorithm must actually learn, not merely not crash.

This is the core CI defence. A smoke test only proves the syntax is right; a learning
test is what catches algorithmic regressions.
"""

import pytest

import oprl
from oprl.algos import ppo, vmpo


def _final_return(algo_mod, cfg, env_id="CartPole-v1", seed=1):
    env = oprl.make_env(env_id, num_envs=cfg.num_envs, seed=seed)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    log = oprl.Logger(sinks=[])
    algo_mod.train(cfg, env, policy, log)
    env.close()
    assert log.ep_returns, "no episode finished -- the training loop is broken"
    return sum(log.ep_returns) / len(log.ep_returns)


def test_ppo_smoke():
    cfg = ppo.PPOConfig(total_steps=4 * 32 * 3, num_envs=4, rollout_len=32,
                        device="cpu", num_epochs=2)
    _final_return(ppo, cfg)


def test_vmpo_smoke():
    cfg = vmpo.VMPOConfig(total_steps=4 * 32 * 3, num_envs=4, rollout_len=32,
                          device="cpu")
    _final_return(vmpo, cfg)


def test_a2c_is_a_ppo_config():
    """A2C is not a separate file, just a set of PPO hyperparameters."""
    cfg = ppo.a2c_config(total_steps=4 * 32 * 2, num_envs=4, rollout_len=32,
                         device="cpu")
    assert cfg.num_epochs == 1 and cfg.num_minibatches == 1
    _final_return(ppo, cfg)


@pytest.mark.slow
def test_ppo_learns_cartpole():
    """PPO must clearly beat a random policy on CartPole (random scores about 20)."""
    cfg = ppo.PPOConfig(total_steps=60_000, num_envs=8, rollout_len=128,
                        device="cpu", num_epochs=4, num_minibatches=4, seed=1)
    ret = _final_return(ppo, cfg)
    assert ret > 100, f"PPO failed to learn CartPole: final mean return {ret:.1f}"


@pytest.mark.slow
def test_vmpo_learns_cartpole():
    cfg = vmpo.VMPOConfig(total_steps=60_000, num_envs=8, rollout_len=128,
                          device="cpu", seed=1)
    ret = _final_return(vmpo, cfg)
    assert ret > 80, f"V-MPO failed to learn CartPole: final mean return {ret:.1f}"

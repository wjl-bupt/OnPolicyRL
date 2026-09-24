"""PPO baseline smoke + learning tests (src/onpolicy/ppo.py).

The policy is the real S0 network from architecture/: `build_policy` assembles it
from the env's spaces and exposes the doc's Policy contract (act / evaluate /
predict_value / dist). Nothing is inherited from the algorithm; PPO no longer
knows Discrete from Box -- the head does.
"""

import pytest
import torch as th

from architecture.actor_critic import PolicyConfig, build_policy
from common.env.adapter import GymVectorAdapter
from common.env.make_env import make_env
from common.logger import Logger
from memory.buffer import RolloutBuffer
from onpolicy.ppo import PPO, PPOConfig
from protocol.buffer import base_schema


def _build(total_steps: int, num_envs: int = 4, rollout_len: int = 32):
    cfg = PPOConfig(num_envs=num_envs, rollout_len=rollout_len,
                    ppo_epoch=2, num_mini_batch=2, total_steps=total_steps,
                    device="cpu", seed=1)
    env = GymVectorAdapter(make_env("CartPole-v1", num_envs=num_envs, seed=1),
                           device=cfg.device)
    schema = base_schema(env.obs_space, env.action_space)
    buf = RolloutBuffer(rollout_len, num_envs, schema, device=cfg.device,
                        spec=None)
    policy = build_policy(env.obs_space, env.action_space, PolicyConfig(lr=3e-4))
    ppo = PPO(env, policy, buf, cfg, log=Logger(sinks=[]))
    return ppo, buf


def test_ppo_smoke_full_chain():
    """collect -> estimate -> update -> log all run; metrics are finite."""
    ppo, buf = _build(total_steps=4 * 32 * 3)
    ppo.train()
    assert ppo.num_updates == 3
    assert th.isfinite(buf.advantages).all()
    assert th.isfinite(buf.returns).all()


def test_ppo_uses_the_declared_sample_tuple():
    """The update consumes protocol `RolloutSamples` -- element names come from
    the declaration, not from dict keys."""
    ppo, buf = _build(total_steps=32 * 4)
    ppo._collect_rollout_()
    ppo._compute_advantages_()
    data = next(iter(buf.sample()))
    # attribute access on the declared NamedTuple is the integration contract
    _ = (data.observations, data.actions, data.old_log_prob, data.old_values,
         data.advantages, data.returns)
    ppo._update_()          # runs without touching dict keys
    assert th.isfinite(buf.advantages).all()


@pytest.mark.slow
def test_ppo_learns_cartpole():
    """A learning-capability test, not a smoke test: PPO must clearly beat a
    random policy (random scores about 20 on CartPole)."""
    ppo, _ = _build(total_steps=40_000)
    ppo.train()
    rew = ppo.log.history("rollout/ep_rew_mean")
    assert rew, "no episode finished -- the training loop is broken"
    assert max(rew) > 50, f"PPO failed to learn: best ep_rew_mean {max(rew):.1f}"

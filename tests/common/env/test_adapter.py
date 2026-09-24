"""GymVectorAdapter tests -- the tensor boundary and the autoreset semantics.

The load-bearing property: under NEXT_STEP autoreset, the step AFTER a
termination is a dummy (valid=False, reward=0, action discarded), and the three
masks never collapse into a `done`.
"""

import numpy as np
import pytest
import torch as th

pytest.importorskip("gymnasium")

import gymnasium as gym

from common.env.adapter import GymVectorAdapter
from protocol.masks import Masks


def _short_cartpole(num_envs: int = 2, steps: int = 5):
    """CartPole with a 5-step time limit -> truncations (and autoreset dummies)
    every few steps. RecordEpisodeStatistics is part of the default stack, so it
    must be present for the adapter to surface episode stats."""
    venv = gym.vector.SyncVectorEnv([
        lambda: gym.wrappers.RecordEpisodeStatistics(
            gym.make("CartPole-v1", max_episode_steps=steps))
        for _ in range(num_envs)])
    return GymVectorAdapter(venv)


def test_tensors_in_and_out_on_device():
    adapter = _short_cartpole()
    try:
        obs = adapter.reset(seed=0)
        assert isinstance(obs, th.Tensor) and obs.dtype == th.float32
        actions = th.zeros(2, dtype=th.long)
        obs, reward, masks, info = adapter.step(actions)
        assert obs.dtype == th.float32 and obs.shape == (2, 4)
        assert reward.dtype == th.float32 and reward.shape == (2,)
        assert masks.terminated.dtype == th.bool
        assert isinstance(masks, Masks)          # the protocol type, three fields
        assert not hasattr(masks, "done"), "no collapsed `done` may ever exist"
        assert isinstance(info, dict)
    finally:
        adapter.close()


def test_autoreset_dummy_steps():
    """The NEXT_STEP contract, per env: valid[t] == not done[t-1] (t=0 follows
    the reset, so it is always valid); a dummy step is never terminal and
    carries no reward. Exact dummy TIMES are env-dependent (episodes may end
    before the time limit), so only the pairing invariant is pinned."""
    adapter = _short_cartpole(num_envs=2, steps=5)
    try:
        obs = adapter.reset(seed=0)
        prev_done = th.zeros(2, dtype=th.bool)
        episodes = []
        saw_dummy = False
        for t in range(20):
            _, reward, masks, info = adapter.step(th.zeros(2, dtype=th.long))
            # the pairing invariant, per env
            assert th.equal(masks.valid, ~prev_done)
            dummy = ~masks.valid
            if dummy.any():
                saw_dummy = True
                assert not masks.terminated[dummy].any()
                assert not masks.truncated[dummy].any()
                assert th.all(reward[dummy] == 0)
            episodes += info["_finished_episodes"]
            prev_done = masks.terminated | masks.truncated
        assert saw_dummy, "5-step limit must produce autoreset dummies"
        # every finished episode has the raw return == its length (reward 1/step)
        assert all(abs(ret - length) < 1e-6 for ret, length in episodes)
        assert episodes, "episodes must be surfaced"
    finally:
        adapter.close()


def test_valid_samples_survive_in_the_buffer():
    """End-to-end with the buffer: dummy steps are excluded from sampling, and
    segments() splits at every truncation."""
    from memory.buffer import RolloutBuffer
    from protocol.buffer import base_schema

    adapter = _short_cartpole(num_envs=2, steps=5)
    buf = RolloutBuffer(20, 2, base_schema(adapter.obs_space, adapter.action_space))
    try:
        obs = adapter.reset(seed=0)
        for _ in range(20):
            buf.write_obs(obs)
            buf.write(action=th.zeros(2, dtype=th.long),
                      logprob=th.zeros(2), value=th.zeros(2))
            obs, reward, masks, _ = adapter.step(th.zeros(2, dtype=th.long))
            buf.write(reward=reward)
            buf.write_masks(masks)
            buf.advance()
        buf.set_bootstrap_value(th.zeros(2))
        buf.set_estimates(th.zeros(20, 2), th.zeros(20, 2))

        n_valid = int(buf.masks.valid.sum())
        n_sampled = sum(mb.actions.shape[0] for mb in buf.sample())
        assert n_sampled == n_valid == 20 * 2 - 6      # six dummy steps excluded
        # segments: per env, 5-step truncated episodes cut at t=5,10,15
        segs = buf.segments()
        assert len([s for s in segs if s[2] - s[1] == 5]) == 6
    finally:
        adapter.close()


def test_dict_obs_adapter():
    pytest.importorskip("minigrid")
    from common.env.minigrid_env import make_minigrid_env

    adapter = GymVectorAdapter(
        make_minigrid_env("MiniGrid-Empty-5x5-v0", num_envs=2, seed=0))
    try:
        obs = adapter.reset(seed=0)
        assert isinstance(obs, dict)
        # numeric leaves become tensors on device; text leaves (mission) pass
        # through as-is -- feeding them to a policy is the dict-obs encoder's job
        assert isinstance(obs["image"], th.Tensor)
        assert isinstance(obs["mission"], (tuple, list, np.ndarray))
        obs, reward, masks, info = adapter.step(th.zeros(2, dtype=th.long))
        assert isinstance(obs["image"], th.Tensor) and reward.shape == (2,)
    finally:
        adapter.close()


def test_action_dtype_conversion():
    """Discrete actions cross as int64 (gymnasium); the adapter casts."""
    adapter = _short_cartpole()
    try:
        adapter.reset(seed=0)
        obs, reward, masks, info = adapter.step(th.tensor([1, 1], dtype=th.int32))
        assert reward.shape == (2,)            # int32 input accepted and cast
    finally:
        adapter.close()

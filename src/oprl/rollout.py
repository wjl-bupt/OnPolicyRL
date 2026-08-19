"""Rollout collection -- the sampling loop shared by every algorithm."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .buffer import RolloutBuffer
from .logger import Logger
from .metrics import Timer
from .types import EnvAdapter, Obs, Policy


@contextmanager
def _null():
    yield


@torch.no_grad()
def collect(
    env: EnvAdapter,
    policy: Policy,
    buf: RolloutBuffer,
    obs: Obs,
    log: Logger,
    timer: Timer | None = None,
    obs_norm=None,
    reward_norm=None,
) -> tuple[Obs, int]:
    """Collect one rollout. Returns (last obs, environment steps consumed).

    Observations are normalized **before** being written to the buffer, so the update
    phase reads already-normalized values. Normalizing in one place only removes the
    chance of the two paths disagreeing.
    """
    buf.reset()
    steps = 0
    for _ in range(buf.T):
        norm_obs = obs_norm(obs) if obs_norm is not None and not isinstance(obs, dict) else obs

        with (timer("fwd") if timer else _null()):
            action, logp, value, _ = policy.act(norm_obs)

        buf.write_obs(norm_obs)
        buf.write(action=action, logprob=logp, value=value)

        with (timer("env") if timer else _null()):
            obs, reward, masks, info = env.step(action)

        for ret, length in info.get("_finished_episodes", []):
            log.add_episode(ret, length)

        stored_reward = (
            reward_norm(reward, masks.terminated, masks.truncated)
            if reward_norm is not None
            else reward
        )
        buf.write(reward=stored_reward)
        buf.write_masks(masks)
        buf.advance()
        steps += env.num_envs

    # bootstrap V(s_T)
    last_obs = obs_norm(obs) if obs_norm is not None and not isinstance(obs, dict) else obs
    buf.set_bootstrap_value(policy.value(last_obs))
    return obs, steps

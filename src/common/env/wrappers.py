"""Shared wrapper pieces and the vectorization helper for the env builders.

Every builder in this package composes its family's default wrapper stack from
these pieces; anything family-specific lives in the family module.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:  # pragma: no cover
    raise ImportError("env builders need gymnasium: uv pip install gymnasium") from e


class FireResetWrapper(gym.Wrapper):
    """Press FIRE once after every reset.

    A few Atari games (Pong, Breakout, ...) stay in a suspended state until FIRE
    is pressed, so the first frames after reset would otherwise be idle. Applied
    opt-in by `make_atari_env(fire_reset=True)`, after AtariPreprocessing and
    before frame stacking.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # 1 = FIRE in the ALE set
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ChannelsFirstWrapper(gym.ObservationWrapper):
    """Transpose image observations [H, W, C] -> [C, H, W].

    MinAtar returns channels-last bool frames; CNN encoders expect channels-first.
    Note MinAtar obs are bool {0,1} and must NOT be rescaled by /255 -- that
    collapses the image (a real bug fixed once already; see git history
    memory/minatar-env-fixes.md).
    """

    def __init__(self, env):
        super().__init__(env)
        s = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.transpose(s.low, (2, 0, 1)),
            high=np.transpose(s.high, (2, 0, 1)),
            dtype=s.dtype,
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


def vectorize(factory, num_envs: int, seed: int | None = None, async_envs: bool = False):
    """Build a Sync/AsyncVectorEnv from a single-env factory.

    When `seed` is given the vector env is reset immediately with per-env seeds
    `seed + i` -- this both fixes the initial state and initializes any stateful
    wrappers (frame stacks). The returned env is ready to step.

    `async_envs=True` requires a picklable (module-level) factory; closures raise
    a pickling error naturally. Sync is the default and what every builder uses.
    """
    make = gym.vector.AsyncVectorEnv if async_envs else gym.vector.SyncVectorEnv
    venv = make([factory] * num_envs)
    if seed is not None:
        venv.reset(seed=seed)
    return venv

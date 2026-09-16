"""MuJoCo env startup -- gymnasium backend only.

Default wrapper stack:

    RecordEpisodeStatistics

and nothing else -- deliberately. The MuJoCo-specific ingredients that actually
matter, obs normalization and reward normalization, are **algorithm-side
normalizers** (frozen during evaluation, checkpointed with the model -- DESIGN_v2
§5.3, DESIGN.md §4.10), not wrappers. Putting them in the env would hide them
from the checkpoint and from eval-time freezing.

Continuous actions; obs is a flat float32 vector.
"""

from __future__ import annotations

import gymnasium as gym

try:
    from common.env.wrappers import vectorize
except ImportError:  # executed from inside src/common/env/
    from wrappers import vectorize


def make_mujoco_env(
    env_id: str = "HalfCheetah-v5",
    num_envs: int = 1,
    seed: int | None = None,
    *,
    async_envs: bool = False,
    use_default_wrappers: bool = True,
    **gym_make_kwargs,
) -> gym.vector.VectorEnv:
    """Build a vectorized MuJoCo env."""

    def _one() -> gym.Env:
        try:
            import mujoco  # noqa: F401
        except ImportError as e:
            raise ImportError("MuJoCo needs mujoco: uv pip install mujoco") from e
        env = gym.make(env_id, **gym_make_kwargs)
        if not use_default_wrappers:
            return env
        return gym.wrappers.RecordEpisodeStatistics(env)

    return vectorize(_one, num_envs, seed, async_envs)


if __name__ == "__main__":
    env = make_mujoco_env("HalfCheetah-v5", num_envs=2, seed=0)
    print("MuJoCo:")
    print("  obs:", env.single_observation_space)
    print("  act:", env.single_action_space)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("  obs batch shape:", obs.shape, "| reward:", reward)
    env.close()

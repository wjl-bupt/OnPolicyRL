"""Classic-control env startup (CartPole / Acrobot / Pendulum / ...).

Default wrapper stack: RecordEpisodeStatistics -- nothing else. Classic control
is the Tier-0 smoke/CI family: the minimal closed loop with no preprocessing.
"""

from __future__ import annotations

import gymnasium as gym

try:
    from common.env.wrappers import vectorize
except ImportError:  # executed from inside src/common/env/
    from wrappers import vectorize


def make_classic_env(
    env_id: str = "CartPole-v1",
    num_envs: int = 1,
    seed: int | None = None,
    *,
    async_envs: bool = False,
    use_default_wrappers: bool = True,
    **gym_make_kwargs,
) -> gym.vector.VectorEnv:
    """Build a vectorized classic-control env."""

    def _one() -> gym.Env:
        env = gym.make(env_id, **gym_make_kwargs)
        if not use_default_wrappers:
            return env
        return gym.wrappers.RecordEpisodeStatistics(env)

    return vectorize(_one, num_envs, seed, async_envs)


if __name__ == "__main__":
    env = make_classic_env("CartPole-v1", num_envs=2, seed=0)
    print("classic control:")
    print("  obs:", env.single_observation_space)
    print("  act:", env.single_action_space)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("  obs batch shape:", obs.shape, "| reward:", reward)
    env.close()

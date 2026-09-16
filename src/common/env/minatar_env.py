"""MinAtar env startup -- gymnasium backend only.

Default wrapper stack:

    RecordEpisodeStatistics
      -> ChannelsFirstWrapper             ([H, W, C] bool -> [C, H, W])

MinAtar frames are 10x10x4 bool {0,1}, channels-last; CNN encoders want
channels-first. No grayscale/resize/stack needed (already compact), and bool obs
must NOT be rescaled by /255 -- that collapses the image (a real bug fixed once
already; see git history memory/minatar-env-fixes.md).

Env ids are registered by the installed `minatar` package itself
(`minatar.register_envs()`; e.g. "MinAtar/Breakout-v0" -- the exact suffix can be
read from `gym.registry` after registration). A previous draft of this file
registered its own ids pointing at `algo.util:BaseEnv`, which does not exist in
this repo; if you later add your own BaseEnv, swap the entry point there.

gymnasium 1.x does not auto-load third-party entry points, so registration
happens lazily inside the builder -- importing this module never pulls the
optional minatar dependency.
"""

from __future__ import annotations

import gymnasium as gym

try:
    from common.env.wrappers import ChannelsFirstWrapper, vectorize
except ImportError:  # executed from inside src/common/env/
    from wrappers import ChannelsFirstWrapper, vectorize


_MINATAR_REGISTERED = False


def _register_minatar() -> None:
    global _MINATAR_REGISTERED
    if _MINATAR_REGISTERED:
        return
    try:
        import minatar
        import minatar.gym  # noqa: F401  (register_envs lives on the submodule)

        # Some minatar versions also expose it at the top level; prefer that.
        register = getattr(minatar, "register_envs", None) or minatar.gym.register_envs
        register()
        _MINATAR_REGISTERED = True
    except ImportError as e:
        raise ImportError("MinAtar needs minatar: uv pip install minatar") from e


def minatar_env_ids() -> list[str]:
    """All MinAtar ids known to gymnasium (after registration)."""
    _register_minatar()
    return sorted(
        s.id for s in gym.registry.values()
        if "minatar" in s.id.lower()
    )


def make_minatar_env(
    env_id: str = "MinAtar/Breakout-v1",
    num_envs: int = 1,
    seed: int | None = None,
    *,
    channels_first: bool = True,
    frame_stack: int = 1,
    async_envs: bool = False,
    use_default_wrappers: bool = True,
    **gym_make_kwargs,
) -> gym.vector.VectorEnv:
    """Build a vectorized MinAtar env with the default wrapper stack."""

    def _one() -> gym.Env:
        _register_minatar()
        env = gym.make(env_id, **gym_make_kwargs)
        if not use_default_wrappers:
            return env
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if channels_first:
            env = ChannelsFirstWrapper(env)
        if frame_stack > 1:
            stack = getattr(gym.wrappers, "FrameStackObservation", None) or getattr(
                gym.wrappers, "FrameStack", None
            )
            env = stack(env, frame_stack)
        return env

    return vectorize(_one, num_envs, seed, async_envs)


if __name__ == "__main__":
    ids = minatar_env_ids()
    print("registered MinAtar ids:", ids)
    breakout = next(i for i in ids if "breakout" in i.lower())
    env = make_minatar_env(breakout, num_envs=2, seed=0)
    print("MinAtar:", breakout)
    print("  obs:", env.single_observation_space)
    print("  act:", env.single_action_space)
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("  obs batch shape:", obs.shape, "| dtype:", obs.dtype, "| reward:", reward)
    env.close()

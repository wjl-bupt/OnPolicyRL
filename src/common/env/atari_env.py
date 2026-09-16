"""Atari (ALE) env startup -- gymnasium backend and envpool backend.

Gymnasium default wrapper stack (the standard DQN/PPO suite, in the order that
matters -- innermost first):

    AtariPreprocessing(noop_max=30, frame_skip=4, grayscale 84x84,
                       terminal_on_life_loss=False)
      -> [FireResetWrapper, opt-in]
      -> RecordEpisodeStatistics          (episode return/length into info)
      -> FrameStackObservation(4)         (defines the final obs shape)

`gym.make` is called with `frameskip=1` so the *preprocessing* frame_skip=4 is the
only frame skip -- v5 ids otherwise default to frameskip=5 and the two would
multiply. Every wrapper is overridable by argument; `use_default_wrappers=False`
returns the bare env (the escape hatch).

envpool backend: envpool applies the standard preprocessing in C++, so its env is
returned as-is -- no wrappers on top. With env_type="gymnasium" its step API is
the batched 5-tuple, matching the gymnasium backend's interface. Details that
differ from the gymnasium stack above (envpool 1.2.7 Atari defaults, read from
the env dump): `use_fire_reset=True`, `episodic_life=False`,
`repeat_action_probability=0.0` (no sticky actions -- ALE v5 defaults to 0.25),
`stack_num=4, frame_skip=4, noop_max=30, gray 84x84`. Interface-wide: actions
are passed as ONE `[batch_size]` int32 (Discrete) / float32 (Box) array per
step (`action_space.sample()` does not have the right shape; a well-shaped
int64 array is silently converted). **Wrong-shaped actions do not raise a
Python exception -- they abort the process from C++** (absl CHECK). Whatever
sits on top (the adapter) must validate shapes before calling step.
"""

from __future__ import annotations

import gymnasium as gym

try:
    from common.env.wrappers import FireResetWrapper, vectorize
except ImportError:  # executed from inside src/common/env/
    from wrappers import FireResetWrapper, vectorize


_ALE_REGISTERED = False


def _register_ale() -> None:
    """`gym.register_envs(ale_py)` is required since ale-py 0.10: gymnasium 1.x
    does not auto-load third-party entry points. Idempotent (re-registering
    makes gymnasium warn on every builder call)."""
    global _ALE_REGISTERED
    if _ALE_REGISTERED:
        return
    try:
        import ale_py
    except ImportError as e:
        raise ImportError("Atari needs ale-py: uv pip install ale-py") from e
    gym.register_envs(ale_py)
    _ALE_REGISTERED = True


def make_atari_env(
    env_id: str = "ALE/Pong-v5",
    num_envs: int = 1,
    seed: int | None = None,
    *,
    noop_max: int = 30,
    frame_skip: int = 4,
    screen_size: int = 84,
    grayscale: bool = True,
    terminal_on_life_loss: bool = False,
    fire_reset: bool = False,
    frame_stack: int = 4,
    async_envs: bool = False,
    use_default_wrappers: bool = True,
    **gym_make_kwargs,
) -> gym.vector.VectorEnv:
    """Build a vectorized Atari env with the default preprocessing stack."""
    _register_ale()

    def _one() -> gym.Env:
        env = gym.make(env_id, frameskip=1, **gym_make_kwargs)
        if not use_default_wrappers:
            return env
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=noop_max,
            frame_skip=frame_skip,
            screen_size=screen_size,
            grayscale_obs=grayscale,
            scale_obs=False,
            terminal_on_life_loss=terminal_on_life_loss,
        )
        if fire_reset:
            env = FireResetWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if frame_stack > 1:
            stack = getattr(gym.wrappers, "FrameStackObservation", None) or getattr(
                gym.wrappers, "FrameStack", None
            )
            env = stack(env, frame_stack)
        return env

    return vectorize(_one, num_envs, seed, async_envs)


def make_atari_envpool(
    env_id: str = "Pong-v5",
    num_envs: int = 8,
    seed: int | None = None,
    **envpool_kwargs,
):
    """Build an envpool Atari env (batched, C++-fast).

    envpool's internal preprocessing already matches the standard stack above, so
    the returned object is used directly: obs [num_envs, 4, 84, 84] uint8, step
    returns the 5-tuple (obs, reward, terminated, truncated, info) with terminal
    obs in `info["terminal_observation"]`. The adapter layer consumes those; no
    wrapper is added here.
    """
    try:
        import envpool
    except ImportError as e:
        raise ImportError(
            "the envpool backend needs the envpool package: uv pip install envpool"
        ) from e
    return envpool.make(
        env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        batch_size=num_envs,
        seed=seed,
        **envpool_kwargs,
    )


if __name__ == "__main__":
    # Smoke check, gymnasium backend:
    env = make_atari_env("ALE/Pong-v5", num_envs=2, seed=0)
    print("gymnasium Atari:")
    print("  obs:", env.single_observation_space)
    print("  act:", env.single_action_space)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("  obs batch shape:", obs.shape, "| reward:", reward)
    env.close()

    # Smoke check, envpool backend (runs only if envpool is installed):
    try:
        import numpy as np

        epool = make_atari_envpool("Pong-v5", num_envs=2, seed=0)
        print("envpool Atari:")
        print("  obs:", epool.observation_space)
        print("  act:", epool.action_space)
        obs, info = epool.reset()
        n = epool.action_space.n
        obs, reward, terminated, truncated, info = epool.step(
            np.random.randint(n, size=2).astype("int32")
        )
        print("  obs batch shape:", obs.shape, "| dtype:", obs.dtype)
    except ImportError as e:
        print("envpool Atari: skipped --", e)

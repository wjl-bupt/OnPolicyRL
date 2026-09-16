"""MiniGrid env startup -- gymnasium backend and envpool backend.

Gymnasium default wrapper stack:

    RecordEpisodeStatistics
      -> [FlatObsWrapper, when obs_mode="flat"]

`obs_mode` picks the observation shape, which decides the policy side:

    "dict" (default)  raw MiniGrid dict obs {image, direction, mission:str} --
                      the path that exercises dict-obs policies (DESIGN.md §6.2).
    "flat"            minigrid's FlatObsWrapper: image + direction flattened into
                      one vector and the mission text tokenized and appended --
                      the common choice for quick MLP baselines.

envpool backend (1.2+ ships MiniGrid natively, C++-fast): returns a batched env
with the same 5-tuple step API. **Observation semantics differ from gymnasium**:

    gymnasium   {"image": (7,7,3) uint8, "direction": int, "mission": str}
    envpool     {"image": (N,7,7,3) uint8, "direction": (N,), "mission": (N,96) uint8}

i.e. envpool pre-tokenizes the mission into a fixed byte array, so the policy
side treats it as input features either way (the "flat" path is implicit).
Actions go in as one `[batch_size]` int32 array per step (a well-shaped int64
array is silently converted, but passing int32 is the contract); **wrong-shaped
actions abort the whole process from C++** -- validate shapes before stepping.
envpool applies no extra wrappers; episode stats come through
`info["elapsed_step"]` / `info["reward"]` per its own convention.
"""

from __future__ import annotations

import gymnasium as gym

try:
    from common.env.wrappers import vectorize
except ImportError:  # executed from inside src/common/env/
    from wrappers import vectorize


def make_minigrid_env(
    env_id: str = "MiniGrid-DoorKey-5x5-v0",
    num_envs: int = 1,
    seed: int | None = None,
    *,
    obs_mode: str = "dict",
    async_envs: bool = False,
    use_default_wrappers: bool = True,
    **gym_make_kwargs,
) -> gym.vector.VectorEnv:
    """Build a vectorized MiniGrid env with the default wrapper stack."""

    def _one() -> gym.Env:
        try:
            import minigrid  # noqa: F401  (registers the MiniGrid-* ids on import)
        except ImportError as e:
            raise ImportError("MiniGrid needs minigrid: uv pip install minigrid") from e
        env = gym.make(env_id, **gym_make_kwargs)
        if not use_default_wrappers:
            return env
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if obs_mode == "flat":
            from minigrid.wrappers import FlatObsWrapper

            env = FlatObsWrapper(env)
        elif obs_mode != "dict":
            raise ValueError(f"obs_mode must be 'dict' or 'flat', got {obs_mode!r}")
        return env

    return vectorize(_one, num_envs, seed, async_envs)


def make_minigrid_envpool(
    env_id: str = "MiniGrid-Empty-5x5-v0",
    num_envs: int = 8,
    seed: int | None = None,
    **envpool_kwargs,
):
    """Build an envpool MiniGrid env (batched, C++-fast).

    See the module docstring for the observation-semantics differences from the
    gymnasium backend. The returned object is used directly -- no wrappers.
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
    for mode in ("dict", "flat"):
        env = make_minigrid_env("MiniGrid-DoorKey-5x5-v0", num_envs=2, seed=0,
                                obs_mode=mode)
        print(f"MiniGrid gymnasium (obs_mode={mode!r}):")
        print("  obs:", env.single_observation_space)
        print("  act:", env.single_action_space)
        obs, _ = env.reset(seed=0)
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
        if isinstance(obs, dict):
            print("  obs keys:", {k: getattr(v, "shape", type(v)) for k, v in obs.items()})
        else:
            print("  obs batch shape:", obs.shape)
        env.close()

    env = make_minigrid_envpool("MiniGrid-Empty-5x5-v0", num_envs=2, seed=0)
    print("MiniGrid envpool:")
    print("  obs:", env.observation_space)
    print("  act:", env.action_space)
    obs, info = env.reset()
    import numpy as np

    obs, reward, terminated, truncated, info = env.step(
        np.random.randint(7, size=2).astype("int32")   # envpool wants int32
    )
    print("  obs keys:", {k: v.shape for k, v in obs.items()})

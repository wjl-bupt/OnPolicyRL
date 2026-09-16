"""Dispatcher: one entry point over the per-family env builders.

    make_env("ALE/Pong-v5", backend="gymnasium", num_envs=8)   -> gym.vector.VectorEnv
    make_env("Pong-v5",     backend="envpool",   num_envs=64)  -> envpool env
    make_env("MiniGrid-DoorKey-5x5-v0", ...)                   -> gymnasium | envpool

Backend availability follows the benchmark matrix: envpool covers **atari and
minigrid**; minatar and mujoco are gymnasium-only. `backend="auto"` picks
gymnasium, the reference path everywhere; envpool is the throughput option for
the two families that ship it.

This layer carries no algorithm knowledge: it never adjusts hyperparameters from
the env id. The framework's adapter (tensors in/out on device) wraps whatever
these builders return.
"""

from __future__ import annotations

import re

_FAMILIES: list[tuple[str, re.Pattern]] = [
    ("atari", re.compile(r"^(ALE/|.*NoFrameskip|.*-ram$)")),
    ("minatar", re.compile(r"^MinAtar/", re.IGNORECASE)),
    ("minigrid", re.compile(r"^(MiniGrid|BabyAI)-")),
    ("mujoco", re.compile(
        r"^(HalfCheetah|Ant|Walker2d|Hopper|Humanoid|Swimmer|Reacher|"
        r"InvertedPendulum|InvertedDoublePendulum|Pusher)")),
]

_ENVPOOL_FAMILIES = {"atari", "minigrid"}


def detect_family(env_id: str) -> str:
    for family, pattern in _FAMILIES:
        if pattern.match(env_id):
            return family
    return "unknown"


def _detect_envpool_family(env_id: str) -> str:
    """envpool task ids are bare ("Pong-v5"), so family membership comes from
    envpool's own task list. Only atari/minigrid are served by this backend."""
    try:
        import envpool
    except ImportError as e:
        raise ImportError("the envpool backend needs: uv pip install envpool") from e
    if env_id in set(envpool.list_all_envs()):
        return "minigrid" if env_id.startswith(("MiniGrid", "BabyAI")) else "atari"
    raise ValueError(
        f"{env_id!r} is not an envpool task; see envpool.list_all_envs(), or use "
        f"the gymnasium backend"
    )


def make_env(
    env_id: str,
    backend: str = "gymnasium",
    num_envs: int = 1,
    seed: int | None = None,
    **kwargs,
):
    """Build an env via the family builder. `backend="auto"` == "gymnasium"."""
    family = detect_family(env_id)
    backend = "gymnasium" if backend == "auto" else backend
    if family == "unknown":
        if backend == "envpool":
            family = _detect_envpool_family(env_id)
        else:
            raise ValueError(
                f"cannot detect the env family of {env_id!r}; call a family builder "
                f"explicitly (make_atari_env / make_minatar_env / make_minigrid_env "
                f"/ make_mujoco_env)"
            )
    if backend not in ("gymnasium", "envpool"):
        raise ValueError(f"backend must be 'gymnasium' or 'envpool', got {backend!r}")
    if backend == "envpool" and family not in _ENVPOOL_FAMILIES:
        raise ValueError(
            f"the envpool backend covers {sorted(_ENVPOOL_FAMILIES)} only "
            f"(requested family: {family!r}); use the gymnasium backend"
        )

    if family == "atari":
        try:
            from common.env.atari_env import make_atari_env, make_atari_envpool
        except ImportError:  # executed from inside src/common/env/
            from atari_env import make_atari_env, make_atari_envpool
        build = make_atari_env if backend == "gymnasium" else make_atari_envpool
    elif family == "minatar":
        try:
            from common.env.minatar_env import make_minatar_env
        except ImportError:
            from minatar_env import make_minatar_env
        build = make_minatar_env
    elif family == "minigrid":
        try:
            from common.env.minigrid_env import make_minigrid_env, make_minigrid_envpool
        except ImportError:
            from minigrid_env import make_minigrid_env, make_minigrid_envpool
        build = make_minigrid_env if backend == "gymnasium" else make_minigrid_envpool
    else:
        try:
            from common.env.mujoco_env import make_mujoco_env
        except ImportError:
            from mujoco_env import make_mujoco_env
        build = make_mujoco_env

    return build(env_id, num_envs=num_envs, seed=seed, **kwargs)


if __name__ == "__main__":
    cases = [
        ("ALE/Pong-v5", "gymnasium"),
        ("Pong-v5", "envpool"),
        ("MinAtar/Breakout-v0", "gymnasium"),
        ("MiniGrid-DoorKey-5x5-v0", "gymnasium"),
        ("MiniGrid-Empty-5x5-v0", "envpool"),
        ("HalfCheetah-v5", "gymnasium"),
    ]
    for env_id, backend in cases:
        env = make_env(env_id, backend=backend, num_envs=2, seed=0)
        space = getattr(env, "single_observation_space", None) or getattr(
            env, "observation_space", None)
        family = detect_family(env_id)
        if family == "unknown" and backend == "envpool":
            family = _detect_envpool_family(env_id)
        print(f"{family:<9} {backend:<9} {env_id:<26} obs={space}")
        if hasattr(env, "close"):
            env.close()

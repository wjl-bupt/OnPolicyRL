"""Assemble the wrapper stack from a preset -- fixed pipeline, still overridable.

The wrapper order is not arbitrary; each step explains why it sits where it does.
"""

from __future__ import annotations

from typing import Any

import torch

from .gym_vec import GymVecAdapter
from .presets import EnvPreset, get_preset


def _apply_wrappers(env, p: EnvPreset):
    """Single-env wrapper stack. **Order matters** -- see the comments per step."""
    import gymnasium as gym

    w = gym.wrappers

    # --- Atari-specific, innermost: these act on raw ALE semantics. ---
    if p.noop_max > 0 and hasattr(w, "AtariPreprocessing"):
        pass  # handled by AtariPreprocessing below
    if p.frame_skip > 1 or p.grayscale or p.resize:
        if hasattr(w, "AtariPreprocessing") and p.noop_max > 0:
            env = w.AtariPreprocessing(
                env,
                noop_max=p.noop_max,
                frame_skip=p.frame_skip,
                screen_size=p.resize or 84,
                grayscale_obs=p.grayscale,
                terminal_on_life_loss=p.episodic_life,
                scale_obs=False,
            )
        else:
            if p.grayscale and hasattr(w, "GrayscaleObservation"):
                env = w.GrayscaleObservation(env)
            if p.resize and hasattr(w, "ResizeObservation"):
                env = w.ResizeObservation(env, (p.resize, p.resize))

    if p.time_limit:
        env = w.TimeLimit(env, max_episode_steps=p.time_limit)

    # --- Stats must sit outside clipping/normalization, or logged returns are
    #     not comparable across runs. ---
    if p.record_stats:
        env = w.RecordEpisodeStatistics(env)

    # --- Reward clipping after stats: affects the training signal only, not logs. ---
    if p.clip_reward:
        env = w.TransformReward(env, lambda r: float(torch.sign(torch.tensor(r))))

    # --- Frame stacking outermost: it defines the final observation shape. ---
    if p.frame_stack > 1:
        stack = getattr(w, "FrameStackObservation", None) or getattr(
            w, "FrameStack", None
        )
        if stack is not None:
            env = stack(env, p.frame_stack)
    return env


def make_env(
    env_id: str,
    num_envs: int = 8,
    device: torch.device | str = "cpu",
    seed: int | None = None,
    preset: Any = "auto",
    extra_wrappers: list | None = None,
    async_envs: bool = False,
    **make_kwargs,
) -> GymVecAdapter:
    """Build a vector env using the fixed preset for its family.

    With `preset='auto'` the family is detected from the env id (Atari / MuJoCo /
    MinAtar / MiniGrid / classic). Any option can be overridden:
    `preset={'preset': 'atari', 'frame_stack': 2}`.

    This function carries **no algorithm knowledge** -- it never adjusts
    hyperparameters based on the env name. Those come only from config/.
    """
    import gymnasium as gym

    p = get_preset(preset, env_id)
    user_wrappers = extra_wrappers or []

    def _factory():
        env = gym.make(env_id, **make_kwargs)
        env = _apply_wrappers(env, p)
        for spec in user_wrappers:   # User wrappers go outermost.
            env = _wrap(env, spec)
        return env

    venv = (
        gym.vector.AsyncVectorEnv([_factory] * num_envs)
        if async_envs
        else gym.vector.SyncVectorEnv([_factory] * num_envs)
    )
    adapter = GymVecAdapter(venv, device=device)
    adapter.preset = p
    if seed is not None:
        adapter.reset(seed=seed)
    return adapter


def _wrap(env, spec):
    """User-supplied wrapper: {from: ./my.py:MyWrapper, ...kwargs}"""
    from ..registry import resolve

    if isinstance(spec, str):
        ctor, params = resolve("env_preset", {"from": spec})
    else:
        ctor, params = resolve("env_preset", spec)
    return ctor(env, **params)

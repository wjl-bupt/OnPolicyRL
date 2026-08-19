"""Environment presets -- one fixed wrapper stack per environment family.

In practice the wrapper stack per family is fixed (Atari needs frame stacking and
reward clipping, MuJoCo needs obs/reward normalization, ...). These are pinned down
here and selected automatically from the env id, **while staying overridable**: any
single option can be changed in the config, or the whole preset replaced.

    env:
      id: ALE/Pong-v5
      preset: auto            # auto | classic | atari | minatar | mujoco | raw
      overrides:              # override any single preset option
        frame_stack: 2
      extra_wrappers:         # append your own, applied outside the preset
        - {from: ./my_wrap.py:MyWrapper, arg: 1}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from ..registry import build, register


@dataclass(frozen=True)
class EnvPreset:
    """Standard preprocessing for one environment family; every option is overridable."""

    name: str = "raw"
    # --- general ---
    record_stats: bool = True      # episode return/length stats (outside normalization)
    time_limit: int | None = None  # None = use the environment's own limit
    # --- image ---
    grayscale: bool = False
    resize: int | None = None      # square side length, e.g. 84
    frame_stack: int = 1
    # --- Atari-specific ---
    noop_max: int = 0              # random no-op starts
    frame_skip: int = 1            # max-and-skip
    episodic_life: bool = False    # life loss ends the episode (training signal only)
    fire_reset: bool = False       # games that need FIRE to start
    clip_reward: bool = False      # sign(r)
    # --- normalization (vector obs; applied by the algorithm side) ---
    suggest_norm_obs: bool = False
    suggest_norm_reward: bool = False
    note: str = ""

    def merged(self, overrides: dict[str, Any] | None) -> EnvPreset:
        if not overrides:
            return self
        known = {f.name for f in self.__dataclass_fields__.values()}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(
                f"unknown env preset options {sorted(unknown)}; available: {sorted(known)}"
            )
        return replace(self, **overrides)


# --------------------------------------------------------------------------- #
#  Fixed presets
# --------------------------------------------------------------------------- #

CLASSIC = EnvPreset(
    name="classic",
    note="CartPole / Acrobot / Pendulum / LunarLander: stats only, no preprocessing",
)

MUJOCO = EnvPreset(
    name="mujoco",
    suggest_norm_obs=True,
    suggest_norm_reward=True,
    note="MuJoCo v5 -- obs/reward normalization is required; without it, little learns",
)

MINATAR = EnvPreset(
    name="minatar",
    note="MinAtar 10x10 -- already compact; the Atari preprocessing stack is unneeded",
)

ATARI = EnvPreset(
    name="atari",
    grayscale=True,
    resize=84,
    frame_stack=4,
    noop_max=30,
    frame_skip=4,
    episodic_life=True,
    fire_reset=True,
    clip_reward=True,
    note="ALE -- the standard preprocessing suite from the DQN/PPO papers",
)

MINIGRID = EnvPreset(
    name="minigrid",
    note="MiniGrid -- dict obs (image+direction+mission); needs a dict-obs policy",
)

RAW = EnvPreset(name="raw", record_stats=True, note="no preprocessing at all")

for _p in (CLASSIC, MUJOCO, MINATAR, ATARI, MINIGRID, RAW):
    register("env_preset", _p.name)(_p)


# --------------------------------------------------------------------------- #
#  Auto-detection
# --------------------------------------------------------------------------- #

_PATTERNS: list[tuple[str, EnvPreset]] = [
    (r"^(ALE/|.*NoFrameskip|.*-ram)", ATARI),
    (r"^MinAtar/", MINATAR),
    (r"^(MiniGrid|BabyAI)-", MINIGRID),
    (
        r"^(HalfCheetah|Ant|Walker2d|Hopper|Humanoid|Swimmer|Reacher|"
        r"InvertedPendulum|InvertedDoublePendulum|Pusher)",
        MUJOCO,
    ),
    (r"^(CartPole|Acrobot|Pendulum|MountainCar|LunarLander|BipedalWalker)", CLASSIC),
]


def detect_preset(env_id: str) -> EnvPreset:
    """Detect the family from the env id. Unknown ids fall back to `raw`, reported
    explicitly rather than silently."""
    for pat, preset in _PATTERNS:
        if re.match(pat, env_id):
            return preset
    return RAW


def get_preset(spec: Any, env_id: str) -> EnvPreset:
    """`spec` may be 'auto', a preset name, or a dict (with `overrides` or `from`)."""
    if spec is None or spec == "auto":
        return detect_preset(env_id)
    if isinstance(spec, EnvPreset):
        return spec
    if isinstance(spec, str):
        return build("env_preset", spec)
    d = dict(spec)
    d.pop("note", None)
    overrides = d.pop("overrides", None)
    base_name = d.pop("preset", None) or d.pop("name", None)
    if "from" in d:
        return build("env_preset", d).merged(overrides)
    base = detect_preset(env_id) if base_name in (None, "auto") else build(
        "env_preset", base_name
    )
    # Remaining dict keys act as direct overrides, avoiding a nested `overrides` block.
    return base.merged({**(overrides or {}), **d})


__all__ = [
    "EnvPreset", "detect_preset", "get_preset",
    "CLASSIC", "MUJOCO", "MINATAR", "ATARI", "MINIGRID", "RAW",
]

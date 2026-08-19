"""Environment adapters and presets.

`EnvAdapter` contract: tensors in and out, already on the device (DESIGN.md §4.2).
`make_env` is an L3 convenience layer applying the fixed wrapper stack per family;
L1/L2 only ever accept an `EnvAdapter` object.
"""

from .factory import make_env
from .gym_vec import GymVecAdapter, TensorEnvAdapter
from .presets import (
    ATARI,
    CLASSIC,
    MINATAR,
    MINIGRID,
    MUJOCO,
    RAW,
    EnvPreset,
    detect_preset,
    get_preset,
)

__all__ = [
    "GymVecAdapter", "TensorEnvAdapter", "make_env",
    "EnvPreset", "detect_preset", "get_preset",
    "CLASSIC", "MUJOCO", "MINATAR", "ATARI", "MINIGRID", "RAW",
]

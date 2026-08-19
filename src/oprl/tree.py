"""map / index / stack over nested observations -- a small stand-in for tensordict.

Keeps the core dependency set minimal (DESIGN.md §4.8).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import torch
from torch import Tensor

from .types import Obs

T = TypeVar("T")


def is_dict_obs(obs: Obs) -> bool:
    return isinstance(obs, dict)


def tree_map(fn: Callable[[Tensor], Tensor], obs: Obs) -> Obs:
    """Apply `fn` to every leaf tensor."""
    if isinstance(obs, dict):
        return {k: fn(v) for k, v in obs.items()}
    return fn(obs)


def tree_index(obs: Obs, idx) -> Obs:
    """Index a subset (accepts ints, slices and tensor indices)."""
    return tree_map(lambda t: t[idx], obs)


def tree_stack(items: Sequence[Obs], dim: int = 0) -> Obs:
    """Stack a sequence of observations."""
    first = items[0]
    if isinstance(first, dict):
        return {k: torch.stack([it[k] for it in items], dim=dim) for k in first}
    return torch.stack(list(items), dim=dim)


def tree_flatten_time(obs: Obs) -> Obs:
    """[T, N, ...] -> [T*N, ...]"""
    return tree_map(lambda t: t.reshape(-1, *t.shape[2:]), obs)


def tree_to(obs: Obs, device: torch.device) -> Obs:
    return tree_map(lambda t: t.to(device), obs)


def tree_shapes(obs: Obs) -> dict[str, tuple[int, ...]]:
    """Used for schema inference and debug printing."""
    if isinstance(obs, dict):
        return {k: tuple(v.shape) for k, v in obs.items()}
    return {"": tuple(obs.shape)}


def obs_leaf_count(obs: Obs) -> int:
    return len(obs) if isinstance(obs, dict) else 1

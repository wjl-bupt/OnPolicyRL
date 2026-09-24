"""Algorithm registry -- the single L4 -> L2 seam (algo_design §1).

The workflow layer treats an algorithm as a black box with exactly one entry:

    train(cfg, env, policy, log, monitor, resume) -> summary dict

`get_algo(name)` returns that entry plus the algorithm's config dataclass (so the
runner can build `cfg` from the experiment file's `algo_params`). This module is
the ONLY place under workflow/ that imports onpolicy/ -- the dependency
discipline that keeps L4 from reaching into an algorithm's internals. Adding
V-MPO later is one `elif` here; nothing else in workflow/ changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

_BUILTIN = ("ppo",)


@dataclass(frozen=True)
class AlgoEntry:
    """What the runner needs to run an algorithm without importing it."""

    train: Callable          # (cfg, env, policy, log, monitor, resume) -> dict
    config_cls: type         # the algorithm's hyperparameter dataclass


def get_algo(name: str) -> AlgoEntry:
    key = str(name).lower()
    if key == "ppo":
        # imported lazily and locally: registry is the seam, so the import lives
        # here and nowhere else in the workflow layer.
        from onpolicy.ppo import PPOConfig, train
        return AlgoEntry(train=train, config_cls=PPOConfig)
    raise ValueError(f"unknown algo {name!r}; built-in algorithms: {list(_BUILTIN)}")

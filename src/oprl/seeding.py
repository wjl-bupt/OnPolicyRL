"""Seeding -- `cfg.seed` must determine the whole run, not just the environment.

Without this, network initialization and minibatch shuffling depend on whatever else has
touched the global torch RNG. That makes a run unreproducible and, worse, makes results
depend on test execution order -- which is exactly how this module came to exist: a DAE
learning test passed in isolation and failed inside the full suite.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU and CUDA).

    Args:
        seed: the run seed.
        deterministic: also force deterministic cuDNN kernels. Off by default because it
            can cost real throughput and some ops have no deterministic implementation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng_state() -> dict:
    """RNG state for checkpointing, so a resumed run continues the same stream."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

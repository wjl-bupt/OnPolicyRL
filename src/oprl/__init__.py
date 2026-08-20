"""oprl -- a lightweight single-agent on-policy deep RL framework.

Core dependencies are torch, numpy, gymnasium and pyyaml. Optional backends
(tensorboard / matplotlib / env suites) are imported lazily, so `import oprl` never
pulls in an environment backend.

Layering (DESIGN.md §3)::

    L2  algos/          ppo.py  vmpo.py           one update rule = one file
    L1  advantages/     GAE / DAE / RVL / GA2E    pluggable estimators
        objectives/     PPO / TR-PPO / SPO / ...  pluggable surrogates
        nets/           encoders + ActorCritic    optional defaults
        envs/           EnvAdapter implementations
        buffer schema rollout logger norm config metrics tree registry
    L0  types.py        protocol definitions, torch only

Discipline: L1 never imports algos/ (enforced by tests/test_architecture.py).
"""

from .advantages import (
    GAE,
    AdvantageEstimator,
    BaseEstimator,
    gae,
    get_estimator,
    register,
)
from .buffer import Minibatch, RolloutBuffer
from .config import Config
from .envs import GymVecAdapter, TensorEnvAdapter, make_env
from .logger import ConsoleSink, JsonlSink, Logger, Sink
from .metrics import Timer, explained_variance
from .nets import ActorCritic
from .norm import ObsNormalizer, RewardNormalizer, RunningMeanStd
from .objectives import Surrogate, get_surrogate
from .rollout import collect
from .schema import Field, Op, Schema, base_schema
from .seeding import load_rng_state, rng_state, seed_everything
from .types import EnvAdapter, Masks, Obs, Policy, RState

__version__ = "0.1.0"

__all__ = [
    # L0 protocols
    "EnvAdapter", "Policy", "Masks", "Obs", "RState",
    # buffer / schema
    "RolloutBuffer", "Minibatch", "Field", "Op", "Schema", "base_schema",
    # advantage
    "gae", "GAE", "AdvantageEstimator", "BaseEstimator", "get_estimator", "register",
    # objective
    "Surrogate", "get_surrogate",
    # networks (optional defaults)
    "ActorCritic",
    # runtime
    "Config", "Timer", "collect", "explained_variance",
    "seed_everything", "rng_state", "load_rng_state",
    "Logger", "Sink", "ConsoleSink", "JsonlSink",
    "ObsNormalizer", "RewardNormalizer", "RunningMeanStd",
    "GymVecAdapter", "TensorEnvAdapter", "make_env",
]

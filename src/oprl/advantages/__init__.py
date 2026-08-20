"""Advantage estimators -- protocol, registry and implementations.

`base.py` holds only the protocol and registry (no concrete algorithm); each
estimator lives in its own file. Adding DAE / RVL / GA2E means adding a file, not
touching `base.py`.
"""

from .base import (
    ESTIMATORS,
    AdvantageEstimator,
    BaseEstimator,
    get_estimator,
    register,
)
from .dae import DAE
from .gae import GAE, MonteCarloAdvantage, gae

__all__ = [
    "AdvantageEstimator", "BaseEstimator", "register", "get_estimator", "ESTIMATORS",
    "gae", "GAE", "MonteCarloAdvantage", "DAE",
]

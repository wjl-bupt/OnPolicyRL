"""Pluggable advantage estimators -- registry plus protocol (inspired by verl).

verl dispatches on `adv_estimator: str` through a registry (gae / grpo / reinforce++ /
rloo ...). We borrow that pattern with two necessary changes:

1. **verl's estimators are pure functions** (token-level reward -> advantage). DAE,
   RVL and GA2E are not: DAE changes the critic's learning target, and GA2E needs the
   policy plus a backward pass during estimation. So the protocol needs more than
   `compute()` -- it also needs `critic_loss()` and `state_dict()`.
2. verl targets LLM RLHF, where grpo/rloo drop the critic via in-group normalization.
   That does not map onto the [T, N] temporal structure of control tasks, so it is
   not ported.

The registry itself is thin: a `register("name")` decorator plus `get_estimator()`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from ..schema import Schema


@runtime_checkable
class AdvantageEstimator(Protocol):
    """Compute advantages and critic targets from a rollout. GAE is one implementation.

    The `policy` / `surrogate` arguments exist for algorithms like GA2E that perform
    gradient alignment during estimation; pure-function implementations ignore them.
    """

    name: str
    # Extra buffer fields this estimator needs (e.g. DAE's full policy distribution).
    extra_fields: Schema
    # Extra policy heads this estimator needs (e.g. DAE's advantage head).
    extra_policy_outputs: tuple[str, ...]

    def compute(
        self, buf, policy=None, surrogate=None, cfg=None
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        """Return (advantages, value_targets, diagnostics)."""
        ...

    def critic_loss(self, policy, mb, cfg) -> tuple[Tensor, dict] | None:
        """Custom critic loss. Returning None means "use the algorithm's default".

        **Why it belongs here**: DAE and RVL change exactly the critic's learning
        target. Splitting advantage computation and critic loss across two places lets
        them disagree -- a silent but fatal class of bug.
        """
        ...

    def on_epoch_start(self, epoch: int, buf, policy=None, surrogate=None, cfg=None) -> None:
        """Hook at PPO epoch boundaries -- used by GA2E's epoch mode to re-select
        lambda; a no-op elsewhere.

        Called explicitly from ppo.py rather than registered as a callback, which
        keeps the "no hidden behaviour" property.
        """
        ...

    def state_dict(self) -> dict: ...
    def load_state_dict(self, d: dict) -> None: ...


class BaseEstimator:
    """Optional convenience base class providing no-op defaults.

    **Inheritance is not required**: any object satisfying the AdvantageEstimator
    protocol works (structural subtyping, DESIGN.md §4.7). This just saves writing
    four no-op methods.
    """

    name = "base"
    extra_fields: Schema = {}
    extra_policy_outputs: tuple[str, ...] = ()

    def critic_loss(self, policy, mb, cfg):
        return None

    def on_epoch_start(self, epoch, buf, policy=None, surrogate=None, cfg=None):
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, d: dict) -> None:
        pass


ESTIMATORS: dict[str, type] = {}


def register(name: str):
    """Register an estimator class, also into the shared component registry."""
    from ..registry import register as _reg

    def deco(cls):
        if name in ESTIMATORS:
            raise KeyError(f"estimator '{name}' is already registered")
        cls.name = name
        ESTIMATORS[name] = cls
        _reg("advantage", name)(cls)
        return cls

    return deco


def get_estimator(spec, cfg=None):
    """Accepts a name, a dict, a class or an instance.

    The dict form supports kwargs and custom implementations::

        advantage: {name: gae, lam: 0.9}
        advantage: {from: ./my_est.py:MyEstimator}
    """
    from ..registry import build

    if isinstance(spec, (str, dict)):
        return build("advantage", spec, cfg=cfg)
    if isinstance(spec, type):
        return build("advantage", spec, cfg=cfg)
    return spec  # already an instance

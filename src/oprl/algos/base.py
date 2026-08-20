"""Algorithm registration -- the sixth component kind.

**Why this file exists.** Every other extension point could be swapped from a config
file by name or by `{from: ./my.py:MyClass}`. Algorithms could not: adding one meant
editing five hardcoded sites in `cli.py` and `algos/__init__.py`. That asymmetry made
the cheapest-sounding claim in DESIGN.md §4.7 ("a new algorithm is just a `train()`
function") the most expensive one in practice.

An algorithm is a **pair**, not a single object: a `train()` function plus the Config
class describing its hyperparameters. So unlike the other kinds it is not instantiated
by `registry.build()`; it is looked up as a record.

    @algo("ppo", PPOConfig, note="...")
    def train(cfg, env, policy, log=None, estimator=None): ...

`defaults` covers the case where two entries share one `train()` and differ only in
hyperparameters -- A2C is PPO with `num_epochs=1, num_minibatches=1, clip_coef=inf`,
which is a config, not a code file (DESIGN.md §4.6).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..registry import register as _register


@dataclass(frozen=True)
class Algo:
    """One algorithm: an update rule plus the Config that parameterizes it."""

    name: str
    train: Callable
    config_cls: type
    # Hyperparameter overrides applied under the dataclass defaults but under YAML/CLI,
    # which is how A2C is expressed without a code file.
    defaults: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def make_config(self, d: dict | None = None, strict: bool = True):
        """Build the config. Precedence: dataclass defaults < `defaults` < `d`."""
        merged = {**self.defaults, **(d or {})}
        return self.config_cls.from_dict(merged, strict=strict)


ALGOS: dict[str, Algo] = {}


def algo(name: str, config_cls: type, *, defaults: dict | None = None, note: str = ""):
    """Register a `train()` function as an algorithm. Returns it unchanged.

    Being a no-op wrapper matters: `ppo.train` stays a plain function that can be
    imported and called directly, with or without the registry.
    """

    def deco(fn: Callable) -> Callable:
        record = Algo(name, fn, config_cls, dict(defaults or {}), note)
        register_algo(record)
        return fn

    return deco


def register_algo(record: Algo) -> Algo:
    """Register an already-built `Algo`. Re-registering the same record is a no-op, so
    module reimport under pytest does not raise."""
    prev = ALGOS.get(record.name)
    if prev is not None:
        if prev == record:
            return record
        raise KeyError(f"algo {record.name!r} is already registered")
    ALGOS[record.name] = record
    _register("algo", record.name)(record)
    return record


def alias(name: str, of: str, *, defaults: dict | None = None, note: str = "") -> Algo:
    """Register a variant sharing another algorithm's `train()` and config class."""
    src = ALGOS[of]
    return register_algo(
        Algo(name, src.train, src.config_cls,
             {**src.defaults, **(defaults or {})}, note)
    )


def registered_algos() -> list[str]:
    return sorted(ALGOS)


def get_algo(spec: str | Algo | dict) -> Algo:
    """Resolve a name, an `Algo`, or a `{from: ./my_algo.py:train}` spec.

    The `from` target may be an `Algo` (fully explicit) or a bare `train()` function. In
    the latter case the config class is read from a `config_cls` attribute if present,
    otherwise it falls back to `PPOConfig` -- whose fields cover the hyperparameters
    shared by every on-policy algorithm. The choice is never hidden: `oprl algos` prints
    the config class of each entry.
    """
    if isinstance(spec, Algo):
        return spec
    if isinstance(spec, dict):
        from ..registry import resolve

        obj, params = resolve("algo", spec)
        if isinstance(obj, Algo):
            return obj
        if not callable(obj):
            raise TypeError(
                f"an algo spec must resolve to a train() function or an Algo, got "
                f"{type(obj).__name__}"
            )
        from .ppo import PPOConfig

        return Algo(
            name=params.get("name") or getattr(obj, "__name__", "custom"),
            train=obj,
            config_cls=getattr(obj, "config_cls", PPOConfig),
            defaults=dict(getattr(obj, "defaults", {})),
            note=str(params.get("note", "")),
        )

    text = str(spec)
    # A path-like or dotted reference is shorthand for the `from` form, so the CLI can
    # take `--algo ./diy/algos/my.py:train` without extra syntax.
    if ":" in text:
        return get_algo({"from": text})
    if text not in ALGOS:
        raise KeyError(
            f"no algo {text!r} registered; available: {registered_algos()}\n"
            "(you can also point at your own file with './my_algo.py:train')"
        )
    return ALGOS[text]


__all__ = ["Algo", "ALGOS", "algo", "alias", "register_algo", "registered_algos",
           "get_algo"]

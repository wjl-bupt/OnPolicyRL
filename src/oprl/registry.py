"""Component registry and config-driven assembly -- the pluggable core.

**Goal**: the config file is the single assembly point. Swapping buffer fields, the
network, the advantage estimator or a loss is declared in YAML, optionally pointing
at a `.py` file you wrote -- no cascading code edits.

A component spec has four forms::

    advantage: gae                              # registered name
    advantage: {name: gae, lam: 0.9}            # registered name + kwargs
    advantage: {from: ./my_est.py:MyEst}        # local .py file
    advantage: {from: mypkg.mod:MyEst, k: 1}    # installed module + kwargs

An object referenced via `from` only has to satisfy the relevant protocol
(structural subtyping): it **inherits nothing** and needs no prior registration.
That is how a user module replaces a framework module.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

# kind -> {name -> class/factory}
REGISTRY: dict[str, dict[str, Any]] = {}

KINDS = ("algo", "advantage", "policy_loss", "value_loss", "encoder", "env_preset")


def register(kind: str, name: str):
    """Register a class or factory under a component kind."""
    if kind not in KINDS:
        raise ValueError(f"unknown component kind {kind!r}; available: {KINDS}")

    def deco(obj):
        table = REGISTRY.setdefault(kind, {})
        if name in table and table[name] is not obj:
            raise KeyError(f"{name!r} is already registered under {kind}")
        table[name] = obj
        if not getattr(obj, "name", None):
            try:
                obj.name = name
            except (AttributeError, TypeError):
                pass
        return obj

    return deco


def registered(kind: str) -> list[str]:
    return sorted(REGISTRY.get(kind, {}))


def _load_from_path(ref: str) -> Any:
    """Resolve `path.py:attr` or `module.path:attr`.

    Local file paths take priority: the common research case is "I wrote a new
    estimator inside my project".
    """
    if ":" not in ref:
        raise ValueError(
            f"`from` must look like './my.py:MyClass' or 'pkg.mod:MyClass', got {ref!r}"
        )
    mod_ref, attr = ref.rsplit(":", 1)
    p = Path(mod_ref)
    if p.suffix == ".py":
        if not p.is_file():
            raise FileNotFoundError(f"no such file: {p}")
        mod_name = f"_oprl_user_{p.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, p)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {p}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_ref)
    if not hasattr(module, attr):
        raise AttributeError(f"{mod_ref} has no attribute {attr!r}")
    return getattr(module, attr)


def resolve(kind: str, spec: Any) -> tuple[Any, dict]:
    """Resolve a spec into (constructor, kwargs). Does not instantiate, because the
    caller may want to inject `cfg`."""
    if spec is None:
        raise ValueError(f"empty spec for {kind}")
    # Already an instance or class: pass through.
    if not isinstance(spec, (str, dict)):
        return spec, {}
    if isinstance(spec, str):
        return _lookup(kind, spec), {}

    params = dict(spec)
    params.pop("note", None)
    ref = params.pop("from", None)
    name = params.pop("name", None) or params.pop("type", None)
    if ref:
        return _load_from_path(ref), params
    if name:
        return _lookup(kind, name), params
    raise ValueError(f"spec for {kind} needs `name`/`type` or `from`: {spec!r}")


def _lookup(kind: str, name: str) -> Any:
    table = REGISTRY.get(kind, {})
    if name not in table:
        raise KeyError(
            f"no {name!r} registered under {kind}; available: {sorted(table)}\n"
            "(you can also point at your own implementation with "
            "{from: ./your_file.py:YourClass})"
        )
    return table[name]


def build(kind: str, spec: Any, **injected) -> Any:
    """Resolve and instantiate. Keys in `injected` are passed only if the constructor
    accepts them, so one spec works for both cfg-aware and plain components.
    """
    ctor, params = resolve(kind, spec)
    if not isinstance(ctor, type) and not callable(ctor):
        return ctor  # already an instance
    if isinstance(ctor, type):
        import inspect

        try:
            accepted = set(inspect.signature(ctor.__init__).parameters)
        except (TypeError, ValueError):
            accepted = set()
        kw = {k: v for k, v in injected.items() if k in accepted}
        kw.update(params)
        return ctor(**kw)
    return ctor(**params)


def describe() -> str:
    """List every available component -- backs `oprl components`."""
    lines = []
    for kind in KINDS:
        names = registered(kind)
        lines.append(f"{kind}:")
        lines.append(f"  {', '.join(names) if names else '(none)'}")
        # Constructor parameters are where a component's own hyperparameters live (see
        # objectives/ppo_family.py), so listing them keeps them discoverable now that they
        # are no longer fields on the algorithm's Config.
        for name in names:
            params = _init_params(REGISTRY[kind][name])
            if params:
                lines.append(f"    {name}: {params}")
    lines.append("")
    lines.append("Custom: write {from: ./your_file.py:YourClass} in the config.")
    return "\n".join(lines)


def _init_params(obj) -> str:
    """Render a component's `__init__` keyword parameters as `name=default` pairs."""
    import inspect

    if not isinstance(obj, type):
        return ""
    try:
        sig = inspect.signature(obj.__init__)
    except (TypeError, ValueError):
        return ""
    out = []
    for p in list(sig.parameters.values())[1:]:  # skip self
        if p.name in ("cfg", "args", "kwargs") or p.kind in (
            p.VAR_POSITIONAL, p.VAR_KEYWORD
        ):
            continue
        out.append(f"{p.name}={p.default!r}" if p.default is not p.empty else p.name)
    return ", ".join(out)

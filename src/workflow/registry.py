"""Discover algorithm modules lazily; extend via oprl.algorithms entry points."""
from dataclasses import dataclass
from importlib import import_module, metadata
from pkgutil import iter_modules
from typing import Callable


def _default_policy(obs, action, network, algo):
    from architecture.actor_critic import build_policy
    return build_policy(obs, action, network)


@dataclass(frozen=True)
class AlgoEntry:
    train: Callable
    config_cls: type
    build_policy: Callable = _default_policy


def _key(name):
    return str(name).lower().replace('-', '')


def _catalog():
    import onpolicy
    modules = {_key(m.name): m.name for m in iter_modules(onpolicy.__path__) if not m.name.startswith(('_', 'test_'))}
    plugins = {}
    for ep in metadata.entry_points(group='oprl.algorithms'):
        key = _key(ep.name)
        if key in modules or key in plugins:
            raise ValueError(f'duplicate algorithm name: {ep.name}')
        plugins[key] = ep
    return modules, plugins


def list_algorithms():
    modules, plugins = _catalog()
    return tuple(sorted(set(modules) | set(plugins)))


def get_algo(name):
    key = _key(name)
    modules, plugins = _catalog()
    if key in plugins:
        provider = plugins[key].load()
        entry = provider if isinstance(provider, AlgoEntry) else provider()
        if not isinstance(entry, AlgoEntry):
            raise TypeError(f'{name}: entry point must provide AlgoEntry')
        return entry
    if key not in modules:
        raise ValueError(f'unknown algo {name!r}; available: {list_algorithms()}')
    mod = import_module('onpolicy.' + modules[key])
    if not isinstance(getattr(mod, 'CONFIG_CLASS', None), type) or not callable(getattr(mod, 'train', None)):
        raise TypeError(f'{mod.__name__} must export CONFIG_CLASS and train')
    return AlgoEntry(mod.train, mod.CONFIG_CLASS, getattr(mod, 'build_policy', _default_policy))

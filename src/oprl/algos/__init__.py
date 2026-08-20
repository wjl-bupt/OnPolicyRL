"""Algorithms (L2) -- one update rule per file.

Importing this package registers the built-ins, so `oprl.algos.base.ALGOS` is populated
and the CLI can dispatch by name without hardcoding one.
"""

from . import ppo, vmpo
from .base import ALGOS, Algo, algo, alias, get_algo, register_algo, registered_algos

__all__ = [
    "ppo", "vmpo",
    "Algo", "ALGOS", "algo", "alias", "register_algo", "registered_algos", "get_algo",
]

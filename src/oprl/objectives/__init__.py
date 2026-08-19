"""Policy and value objectives -- both pluggable.

**Why these live in L1 rather than algos/**: they are primitives *used by* algorithms,
not algorithms themselves. Putting them in algos/ would break the "one update rule =
one file" criterion (DESIGN.md §4.6).
"""

from .ppo_family import SURROGATES, Surrogate, get_surrogate
from .value_loss import ClippedValueLoss, HuberValueLoss, MSEValueLoss


def get_value_loss(spec):
    """Accepts a name, a dict or an object. Defaults to `clipped` (original PPO)."""
    from ..registry import build

    if spec is None:
        spec = "clipped"
    if not isinstance(spec, (str, dict)):
        return spec
    return build("value_loss", spec)


__all__ = [
    "Surrogate", "get_surrogate", "SURROGATES", "get_value_loss",
    "ClippedValueLoss", "MSEValueLoss", "HuberValueLoss",
]

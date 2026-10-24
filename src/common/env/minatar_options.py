"""MinAtar-specific environment defaults, shared by both simulation backends."""
import math

DEFAULT_STICKY_ACTION_PROB = 0.1


def sticky_probability(value=DEFAULT_STICKY_ACTION_PROB):
    probability = float(value)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('sticky_action_prob must be finite and in [0, 1]')
    return probability

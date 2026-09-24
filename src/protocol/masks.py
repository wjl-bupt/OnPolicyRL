"""The three rollout masks (DESIGN.md §4.1) -- the framework's most important
correctness type.

The buffer stores `terminated` / `truncated` / `valid` and there is no API that
returns a collapsed `done`:

* `terminated` cuts the value bootstrap (a true MDP end has no continuation);
* `truncated` KEEPS the bootstrap (a time-limited state still has value) but
  cuts the advantage recursion;
* `valid=False` marks autoreset dummy steps -- excluded from sampling, so the
  algorithm layer never needs to know autoreset exists.

Collapsing any two of these is the most widespread on-policy bug.
"""

from typing import NamedTuple

from torch import Tensor


class Masks(NamedTuple):
    terminated: Tensor   # [T, N] bool -- cuts the bootstrap
    truncated: Tensor    # [T, N] bool -- keeps the bootstrap
    valid: Tensor        # [T, N] bool -- False = autoreset dummy step

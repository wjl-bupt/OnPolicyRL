"""Sampling-side declarations: how the buffer is cut AND what a sample contains.

Three declaration layers, each answering one question:

1. **Element level** -- `RolloutSamples` + `SampleMapping`: the sampled object is
   a NamedTuple with named, statically-typed elements (SB3's `RolloutBufferSamples`
   pattern), and a declared mapping from each element back to its buffer source.
   A minibatch is never an anonymous dict.
2. **Operation level** -- `SampleOp`: how each field is cut when sampled
   (flatten / whole / sequence). One field, one op; the op lives on the field
   (see buffer.Field).
3. **Spec level** -- `MinibatchSpec`: how many minibatches, flat vs sequence,
   whether autoreset dummy steps are dropped.

Discipline: the operators here are a **finite, predefined set**, not user
callbacks -- the moment a schema can execute arbitrary code we are back to an
opaque config black box. A genuinely new operator is a framework change made
together with its first user -- never speculatively.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

from torch import Tensor

# --------------------------------------------------------------------------- #
#  1. element level: the sampled object
# --------------------------------------------------------------------------- #


class RolloutSamples(NamedTuple):
    """The element-level view of one sampled minibatch.

    Element names follow SB3's `RolloutBufferSamples` so anyone who has read an
    SB3 buffer reads this without translation. `observations` is a tensor for
    flat obs and a dict of tensors for dict obs (MiniGrid), mirroring the buffer.

    Estimators that need more per-sample fields (a stored distribution, an
    anchor) get an extended class from `sample_class()` below -- declared at
    setup, never grown at runtime.
    """

    observations: "Tensor | dict[str, Tensor]"
    actions: Tensor
    old_values: Tensor
    old_log_prob: Tensor
    advantages: Tensor
    returns: Tensor


_CORE_ELEMENTS: tuple[str, ...] = tuple(RolloutSamples.__annotations__)


class TrajectorySamples(NamedTuple):
    """The element-level view of one **trajectory batch** (mode="trajectory").

    The six core elements are the same as `RolloutSamples`, but their tensors
    are the trajectories of the batch **concatenated along the sample axis**;
    `lengths` recovers per-trajectory views via ``split(lengths)``. `last_values`
    is V(s_end) per trajectory, already zeroed where the trajectory ended on a
    true termination (nothing to bootstrap from).

    Fixed shapes do not exist here -- a batch is a variable number of variable
    length trajectories, which is the point of the mode.
    """

    observations: "Tensor | dict[str, Tensor]"
    actions: Tensor
    old_values: Tensor
    old_log_prob: Tensor
    advantages: Tensor
    returns: Tensor
    lengths: Tensor       # [n_traj] int64
    last_values: Tensor   # [n_traj] float32


def sample_class(extra: "tuple[str, ...] | list[str]",
                 base: type = RolloutSamples) -> type:
    """Build an extended sample class: `base`'s elements plus `extra`
    element-level fields (in order), e.g. ``sample_class(("probs",))`` for an
    estimator that stores the full action distribution per sample, or
    ``sample_class(("probs",), base=TrajectorySamples)`` in trajectory mode.

    Names must be valid identifiers and must not collide with the base. The
    result is a real NamedTuple subclass: attribute access, unpacking and
    ``_asdict()`` all work. IDE support inside a run is best when the estimator
    assigns the return value to a module-level constant, so the extended shape
    is declared once and read everywhere.
    """
    core_names = tuple(base.__annotations__)
    core = [n for n in core_names if n != "observations"]
    bad = [n for n in extra if not n.isidentifier() or n in core_names]
    if bad:
        raise ValueError(
            f"invalid extra sample elements {bad}; they must be identifiers and "
            f"must not collide with the base elements {core_names}"
        )
    ordered: list[tuple[str, Any]] = [(n, Any) for n in core_names]
    ordered += [(n, Any) for n in extra]
    return NamedTuple(base.__name__, ordered)  # type: ignore[return-value]


@dataclass(frozen=True)
class SampleMapping:
    """Declared mapping: sample element -> its source in the buffer.

    `advantages` / `returns` are produced by the estimator (not stored schema
    fields) and must map to themselves; the rest name schema fields.
    `observations` maps to the obs group: a tensor field "obs" or the expanded
    "obs.*" sub-fields.

    Validated here so a misspelling dies at setup, not mid-run.
    """

    observations: str = "obs"
    actions: str = "action"
    old_log_prob: str = "logprob"
    old_values: str = "value"
    advantages: str = "advantages"   # estimator output slot
    returns: str = "returns"         # estimator output slot

    def __post_init__(self) -> None:
        for name in ("advantages", "returns"):
            src = getattr(self, name)
            if src != name:
                raise ValueError(
                    f"sample element {name!r} must map to itself (estimator "
                    f"output slot), got {src!r}"
                )
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, str) or not v:
                raise ValueError(f"mapping for {f.name!r} must be a non-empty string")

    def sources(self) -> dict[str, str]:
        """element -> buffer source, as a plain dict."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


def validate_against_schema(mapping: SampleMapping, schema: dict[str, Any],
                            extra: "tuple[str, ...] | list[str]" = ()) -> None:
    """Cross-check the sample declaration against the buffer schema: every mapped
    source and every extra element must exist in the schema (estimator output
    slots excepted). Call once at buffer construction."""
    for element, src in mapping.sources().items():
        if element in ("advantages", "returns"):
            continue
        if src not in schema:
            raise ValueError(
                f"sample element {element!r} maps to buffer field {src!r}, which "
                f"is not in the schema {sorted(schema)}"
            )
    missing = [e for e in extra if e not in schema]
    if missing:
        raise ValueError(
            f"extra sample elements {missing} are not declared in the buffer "
            f"schema {sorted(schema)}"
        )


# --------------------------------------------------------------------------- #
#  2. operation level: how a field is cut
# --------------------------------------------------------------------------- #


class SampleOp(str, Enum):
    """How a field is cut when the buffer yields samples.

    FLATTEN    [T, N, ...] -> [T*N, ...] then index (the default; feed-forward
               minibatches).
    WHOLE      never cut -- passed through uncut (episode boundary indices,
               whole-rollout anchors).
    SEQUENCE   keep the time axis ([L, B, ...]) -- for sequence minibatches
               (recurrent policies, telescoping objectives).
    """

    FLATTEN = "flatten"
    WHOLE = "whole"
    SEQUENCE = "sequence"


# --------------------------------------------------------------------------- #
#  3. spec level: how the rollout is split into minibatches
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MinibatchSpec:
    """How one update iteration splits the rollout into minibatches (S7 in
    algo_design §9.2 -- a framework stage, parameterized here, never redefined
    per algorithm).

    mode "flat"       shuffle [T*N] valid samples into `num_minibatches` equal
                      minibatches (feed-forward path).
    mode "trajectory" pack **whole trajectories** (the buffer's contiguous
                      segments) into batches: trajectories are shuffled, then
                      drawn one by one until the batch holds about
                      `trajectory_frames` samples ("≈ m"; a batch is always ≥
                      the target -- a trajectory is never cut). With
                      `trajectory_frames=None` the target is derived from
                      `num_minibatches` as total_frames / num_minibatches, so
                      the rollout still ends up in roughly that many batches.
                      A trailing batch holding fewer than `min_trajectories`
                      trajectories is discarded (0 disables dropping). Time is
                      preserved inside each trajectory; autoreset dummy steps
                      belong to no trajectory and are excluded by construction.

    drop_invalid is the flat-mode switch; trajectory mode excludes invalid
    steps structurally, so it is not consulted there.

    Fixed-length [L, B] blocks with recurrent states (the old "sequence" mode)
    are not declared yet -- recurrent policies come later.
    """

    num_minibatches: int = 4
    mode: str = "flat"                     # "flat" | "trajectory"
    drop_invalid: bool = True
    trajectory_frames: int | None = None   # ≈ m samples per batch (trajectory mode)
    min_trajectories: int = 2              # drop a smaller trailing batch; 0 keeps it

    def __post_init__(self) -> None:
        if self.mode not in ("flat", "trajectory"):
            raise ValueError(
                f"mode must be 'flat' or 'trajectory', got {self.mode!r} "
                "(fixed-length recurrent blocks are not declared yet)"
            )
        if self.num_minibatches < 1:
            raise ValueError(f"num_minibatches must be >= 1, got {self.num_minibatches}")
        if self.mode == "flat" and (self.trajectory_frames is not None
                                    or self.min_trajectories != 2):
            raise ValueError(
                "trajectory_frames / min_trajectories are trajectory-mode "
                "options; set mode='trajectory' or remove them"
            )
        if self.trajectory_frames is not None and self.trajectory_frames < 1:
            raise ValueError(f"trajectory_frames must be >= 1, got {self.trajectory_frames}")
        if self.min_trajectories < 0:
            raise ValueError(f"min_trajectories must be >= 0, got {self.min_trajectories}")

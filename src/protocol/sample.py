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


def sample_class(extra: "tuple[str, ...] | list[str]") -> type:
    """Build an extended sample class: the six core elements plus `extra`
    element-level fields (in order), e.g. ``sample_class(("probs",))`` for an
    estimator that stores the full action distribution per sample.

    Names must be valid identifiers and must not collide with the core. The
    result is a real NamedTuple subclass: attribute access, unpacking and
    ``_asdict()`` all work. IDE support inside a run is best when the estimator
    assigns the return value to a module-level constant, so the extended shape
    is declared once and read everywhere.
    """
    core = [n for n in _CORE_ELEMENTS if n != "observations"]
    bad = [n for n in extra if not n.isidentifier() or n in core or n == "observations"]
    if bad:
        raise ValueError(
            f"invalid extra sample elements {bad}; they must be identifiers and "
            f"must not collide with the core elements {_CORE_ELEMENTS}"
        )
    ordered: list[tuple[str, Any]] = [(n, Any) for n in _CORE_ELEMENTS]
    ordered += [(n, Any) for n in extra]
    return NamedTuple("RolloutSamples", ordered)  # type: ignore[return-value]


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

    mode "flat"     shuffle [T*N] valid samples into equal minibatches.
    mode "sequence" cut by (env, time-block), preserving time within a block;
                    fields whose SampleOp is FLATTEN cannot appear in sequence
                    samples -- reserved for recurrent policies.

    drop_invalid: autoreset dummy steps are excluded from every minibatch here,
    so the algorithm layer never needs to know autoreset exists.
    """

    num_minibatches: int = 4
    mode: str = "flat"           # "flat" | "sequence"
    drop_invalid: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("flat", "sequence"):
            raise ValueError(f"mode must be 'flat' or 'sequence', got {self.mode!r}")
        if self.num_minibatches < 1:
            raise ValueError(f"num_minibatches must be >= 1, got {self.num_minibatches}")

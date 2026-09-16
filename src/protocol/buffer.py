"""Buffer-side declarations: what fields a rollout buffer stores.

One field, one definition: `Field` carries BOTH the write-side op (here) and
the sample-side op (SampleOp, defined in sample.py). Splitting a field across
two declaration sites would let the two views drift apart.

This is a protocol layer: it declares layout and validates it, and it stays
torch-only (duck-typed spaces, no gymnasium import). The allocation / write /
sample machinery that *consumes* these declarations lives in the buffer
implementation (common/), not here.

Discipline (matching sample.py): write operators are a **finite, predefined
set** -- STORE / ACCUMULATE / LAST -- not user callbacks. A field's value is
written at exactly one sanctioned point during collection (`observe` for
component-declared fields, the rollout loop for core fields); anything more
custom than the three operators does not belong in a schema.

Why not an actual .proto file: protobuf's type system cannot express a torch
dtype, and its scalars/enum/model (pure data, no validation, no derived
properties) would still need a Python translation layer before a single tensor
could be allocated -- an intermediate representation with no consumer, plus a
protoc codegen step and a runtime dependency. Our problem is in-process tensor
memory layout, not cross-language serialization. If buffer data ever needs to
cross a process or language boundary, export *from* this declaration is
mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from .sample import SampleOp


class WriteOp(str, Enum):
    """How a write into the current slot behaves.

    STORE       assign as-is (one write per step; the default).
    ACCUMULATE  add into the current slot (several writes per step, e.g. shaped
                rewards contributed by separate attachments).
    LAST        keep only the final write within the step.
    """

    STORE = "store"
    ACCUMULATE = "accumulate"
    LAST = "last"


@dataclass(frozen=True)
class Field:
    """Describes one buffer field for a single env at a single step (the [T, N]
    prefix is implicit and never written here).

    Examples::

        Field((4,))                                        # vector field
        Field((), torch.float32, extra_step=True)          # value, +V(s_T) slot
        Field((), torch.long, sample_op=SampleOp.WHOLE)    # never cut

    Unknown-typing mistakes fail here, at construction -- not three hours into a
    training run.
    """

    shape: tuple[int, ...] = ()
    dtype: torch.dtype = torch.float32
    # One extra slot on the time axis: `value` keeps room for the bootstrap V(s_T).
    extra_step: bool = False
    write_op: WriteOp = WriteOp.STORE
    sample_op: SampleOp = SampleOp.FLATTEN
    doc: str = ""

    @property
    def per_sample(self) -> bool:
        """Every operator except WHOLE is cut per sample."""
        return self.sample_op is not SampleOp.WHOLE


BufferSchema = dict[str, Field]


def _space_shape(space) -> tuple[int, ...]:
    return tuple(space.shape) if space.shape is not None else ()


def _dtype(space) -> torch.dtype:
    return torch.uint8 if np.dtype(space.dtype) == np.uint8 else torch.float32


def _action_dtype(space) -> torch.dtype:
    name = type(space).__name__
    if name in ("Discrete", "MultiDiscrete", "MultiBinary"):
        return torch.long
    return torch.float32


def _action_shape(space) -> tuple[int, ...]:
    name = type(space).__name__
    if name == "Discrete":
        return ()
    if name == "MultiDiscrete":
        return (len(space.nvec),)
    return _space_shape(space)


def base_schema(obs_space, act_space) -> BufferSchema:
    """The core fields every on-policy buffer carries, inferred from the spaces
    (duck-typed: anything with .shape / .n works -- no gymnasium import here).

    The three masks are declared, never collapsed into a `done`: the value
    bootstrap is cut only by `terminated`, the recursion by either, and the
    autoreset dummy step is excluded at sample time (doc: DESIGN.md §4.1).
    """
    schema: BufferSchema = {}
    if type(obs_space).__name__ == "Dict":
        for k, sub in obs_space.spaces.items():
            schema[f"obs.{k}"] = Field(_space_shape(sub), _dtype(sub), doc=f"obs/{k}")
    else:
        schema["obs"] = Field(_space_shape(obs_space), _dtype(obs_space), doc="observation")
    schema["action"] = Field(_action_shape(act_space), _action_dtype(act_space),
                             doc="sampled action")
    schema["logprob"] = Field((), torch.float32, doc="log pi_old(a|s)")
    schema["reward"] = Field((), torch.float32, doc="possibly normalized")
    schema["value"] = Field((), torch.float32, extra_step=True,
                            doc="V(s_t); extra slot = V(s_T) bootstrap")
    schema["terminated"] = Field((), torch.bool, doc="cuts the bootstrap")
    schema["truncated"] = Field((), torch.bool, doc="keeps the bootstrap")
    schema["valid"] = Field((), torch.bool, doc="False = autoreset dummy step")
    return schema


def merge(*schemas: BufferSchema) -> BufferSchema:
    """Merge schemas; collisions raise instead of silently overwriting."""
    out: BufferSchema = {}
    for s in schemas:
        dup = set(s) & set(out)
        if dup:
            raise ValueError(f"schema field collision: {sorted(dup)}")
        out.update(s)
    return out


def describe_schema(schema: BufferSchema) -> str:
    """The declaration is printable, so "what is in this buffer" is answerable."""
    lines = ["BufferSchema:"]
    for name, f in sorted(schema.items()):
        steps = "T+1" if f.extra_step else "T"
        ops = "" if (f.write_op is WriteOp.STORE and f.sample_op is SampleOp.FLATTEN) \
            else f"  [{f.write_op.value}/{f.sample_op.value}]"
        doc = f"  # {f.doc}" if f.doc else ""
        lines.append(
            f"  {name:<12} [{steps}, N, {list(f.shape)}] "
            f"{str(f.dtype).replace('torch.', ''):<8}{ops}{doc}"
        )
    return "\n".join(lines)

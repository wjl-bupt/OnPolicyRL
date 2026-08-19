"""Declarative buffer field schema (DESIGN.md §4.3).

In the reference implementation each algorithm ships its own `buffer.py` (6 files,
~1978 lines) even though the only real difference is a handful of extra fields.
Here an algorithm declares just its increment; allocation, writing and minibatch
slicing are handled once.

Discipline: `Field` only ever describes memory layout. Callbacks such as
`transform=` / `init_fn=` are **forbidden** -- the moment a schema can execute code
we are back to an opaque config black box.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


class Op(str, Enum):
    """**Declarative operators** applied to a field on write and on sample.

    This is the "operator form" of buffer customization: changing buffer behaviour
    means changing one enum value in the schema, not editing `write()` / `sample()`.

    Same discipline as `Field`: operators are a **finite, predefined set**, not user
    callbacks. Allowing arbitrary functions would let a schema execute code. Adding a
    new operator means adding one enum member plus one branch in the buffer, which
    stays reviewable and exhaustive.
    """

    # --- write side ---
    STORE = "store"            # write as-is (default)
    ACCUMULATE = "accumulate"  # add into the current slot (multiple writes per step)
    LAST = "last"              # keep only the final write

    # --- sample side ---
    FLATTEN = "flatten"        # [T,N,...] -> [T*N,...] then index (default)
    WHOLE = "whole"            # pass through uncut (e.g. episode boundary indices)
    SEQUENCE = "sequence"      # keep the time axis [L,B,...] (for SequenceSampler)


@dataclass(frozen=True)
class Field:
    """Describes one field for a single env at a single step (no [T, N] prefix).

    Example::

        Field((4,), torch.float32)                   # plain vector field
        Field((), torch.float32, extra_step=True)    # value, keeps room for V(s_T)
        Field((), torch.long, sample_op=Op.WHOLE)    # episode boundaries, never cut
    """

    shape: tuple[int, ...] = ()
    dtype: torch.dtype = torch.float32
    # One extra slot on the time axis: `value` needs V(s_T) for bootstrapping.
    extra_step: bool = False
    # Declarative operators.
    write_op: Op = Op.STORE
    sample_op: Op = Op.FLATTEN
    # Only for dict-obs sub-fields: remembers the original key for reconstruction.
    obs_key: str | None = None
    # Human-readable description -- the schema doubles as documentation.
    doc: str = ""

    @property
    def per_sample(self) -> bool:
        """Every operator except WHOLE slices per sample."""
        return self.sample_op is not Op.WHOLE


Schema = dict[str, Field]


def merge(*schemas: Schema) -> Schema:
    """Merge schemas, raising on field collisions rather than silently overwriting."""
    out: Schema = {}
    for s in schemas:
        dup = set(s) & set(out)
        if dup:
            raise ValueError(f"schema field collision: {sorted(dup)}")
        out.update(s)
    return out


def _space_shape(space) -> tuple[int, ...]:
    return tuple(space.shape) if space.shape is not None else ()


def action_dtype(space) -> torch.dtype:
    """Discrete actions are stored as long, continuous ones as float32."""
    name = type(space).__name__
    if name in ("Discrete", "MultiDiscrete", "MultiBinary"):
        return torch.long
    return torch.float32


def action_shape(space) -> tuple[int, ...]:
    name = type(space).__name__
    if name == "Discrete":
        return ()
    if name == "MultiDiscrete":
        return (len(space.nvec),)
    return _space_shape(space)


def obs_fields(obs_space) -> Schema:
    """Derive obs fields from the observation space; dict obs expand to obs.<key>."""
    if type(obs_space).__name__ == "Dict":
        out: Schema = {}
        for k, sub in obs_space.spaces.items():
            out[f"obs.{k}"] = Field(
                _space_shape(sub), _np_to_torch_dtype(sub.dtype), obs_key=k
            )
        return out
    return {"obs": Field(_space_shape(obs_space), _np_to_torch_dtype(obs_space.dtype))}


def _np_to_torch_dtype(dt) -> torch.dtype:
    dt = np.dtype(dt)
    if dt == np.uint8:
        return torch.uint8
    if dt in (np.int64, np.int32):
        return torch.long
    return torch.float32


def base_schema(obs_space, act_space) -> Schema:
    """Core fields -- inferred from the spaces, zero configuration required."""
    sch: Schema = {}
    sch.update(obs_fields(obs_space))
    sch["action"] = Field(action_shape(act_space), action_dtype(act_space),
                          doc="sampled action")
    sch["logprob"] = Field((), torch.float32, doc="log prob under pi_old")
    sch["reward"] = Field((), torch.float32, doc="possibly normalized")
    sch["value"] = Field((), torch.float32, extra_step=True,
                         doc="V(s_t); extra slot holds V(s_T) for bootstrapping")
    sch["terminated"] = Field((), torch.bool, doc="true termination; cuts bootstrap")
    sch["truncated"] = Field((), torch.bool, doc="time limit; bootstrap is kept")
    sch["valid"] = Field((), torch.bool, doc="False = autoreset dummy step")
    return sch


# --------------------------------------------------------------------------- #
#  Declaring buffer fields from configuration
# --------------------------------------------------------------------------- #
#
# Why not an actual .proto file: protobuf's type system cannot express a torch dtype
# plus a multi-dimensional shape (`repeated float` is one-dimensional), and it would
# add a protoc codegen step and a runtime dependency. Protobuf solves cross-language,
# cross-process serialization; our problem is in-process tensor memory layout.
#
# The underlying request -- putting the structure definition in the config file -- is
# right, and this is where it lands:
#
#     buffer:
#       extra:
#         policy:   {shape: [n_actions], dtype: float32, doc: "DAE needs full dist"}
#         ep_start: {dtype: int64, sample_op: whole}
#         adv_raw:  {shape: [], dtype: float32, extra_step: true}

_DTYPES = {
    "float32": torch.float32, "float": torch.float32, "f32": torch.float32,
    "float64": torch.float64, "double": torch.float64,
    "float16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64, "long": torch.int64, "int": torch.int64,
    "int32": torch.int32, "int16": torch.int16,
    "uint8": torch.uint8, "byte": torch.uint8,
    "bool": torch.bool,
}


def parse_dtype(spec) -> torch.dtype:
    if isinstance(spec, torch.dtype):
        return spec
    key = str(spec).replace("torch.", "").lower()
    if key not in _DTYPES:
        raise ValueError(f"unknown dtype {spec!r}; available: {sorted(_DTYPES)}")
    return _DTYPES[key]


def parse_field(spec: dict | Field, symbols: dict[str, int] | None = None) -> Field:
    """Turn one field declaration from the config into a `Field`.

    `shape` accepts symbolic names (e.g. n_actions / obs_dim) resolved via `symbols`,
    so configs need not hard-code environment-dependent dimensions.
    """
    if isinstance(spec, Field):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"field declaration must be a mapping, got {type(spec).__name__}")
    d = dict(spec)
    raw_shape = d.pop("shape", ())
    if isinstance(raw_shape, (int, str)):
        raw_shape = [raw_shape]
    shape = tuple(_resolve_dim(x, symbols or {}) for x in raw_shape)
    field = Field(
        shape=shape,
        dtype=parse_dtype(d.pop("dtype", "float32")),
        extra_step=bool(d.pop("extra_step", False)),
        write_op=Op(str(d.pop("write_op", "store")).lower()),
        sample_op=Op(str(d.pop("sample_op", "flatten")).lower()),
        doc=str(d.pop("doc", "")),
    )
    if d:  # Unknown keys are an error -- never silently ignore a typo.
        raise ValueError(
            f"unknown field attributes {sorted(d)}; "
            "available: shape, dtype, extra_step, write_op, sample_op, doc"
        )
    return field


def _resolve_dim(x, symbols: dict[str, int]) -> int:
    if isinstance(x, int):
        return x
    key = str(x)
    if key not in symbols:
        raise ValueError(
            f"cannot resolve symbol {key!r} in shape; available: {sorted(symbols)} "
            "(or write a plain integer)"
        )
    return int(symbols[key])


def schema_from_config(spec: dict | None, symbols: dict[str, int] | None = None) -> Schema:
    """Turn the `buffer.extra` block from the config into a `Schema`."""
    if not spec:
        return {}
    return {k: parse_field(v, symbols) for k, v in spec.items()}


def env_symbols(obs_space, act_space) -> dict[str, int]:
    """Symbol table for shapes, so configs can say `n_actions` instead of a literal."""
    import numpy as np

    sym: dict[str, int] = {}
    if type(act_space).__name__ == "Discrete":
        sym["n_actions"] = int(act_space.n)
    elif act_space.shape:
        sym["n_actions"] = int(np.prod(act_space.shape))
        sym["act_dim"] = sym["n_actions"]
    if getattr(obs_space, "shape", None):
        sym["obs_dim"] = int(np.prod(obs_space.shape))
    return sym

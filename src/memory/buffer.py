# -*- encoding: utf-8 -*-
'''
@File :buffer.py
@Created-Time :2026-09-16 15:45:03
@Author  :june
@Description   : Rollout Buffer (on-policy) -- consumes src/protocol declarations.
@Modified-Time :2026-09-16 15:45:03
'''

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, NamedTuple

import torch
from torch import Tensor

try:
    from protocol.buffer import BufferSchema, WriteOp, describe_schema
    from protocol.sample import (
        MinibatchSpec,
        RolloutSamples,
        SampleMapping,
        SampleOp,
        sample_class,
        validate_against_schema,
    )
except ImportError:  # executed from inside src/memory/ without `src` on sys.path
    from protocol.buffer import BufferSchema, WriteOp, describe_schema  # type: ignore
    from protocol.sample import (  # type: ignore
        MinibatchSpec,
        RolloutSamples,
        SampleMapping,
        SampleOp,
        sample_class,
        validate_against_schema,
    )


class Masks(NamedTuple):
    """The three rollout masks (DESIGN.md §4.1). The canonical definition will
    move to src/protocol when the env adapter lands; the buffer only ever
    exposes them together -- no API returns a collapsed `done`."""

    terminated: Tensor
    truncated: Tensor
    valid: Tensor


class RolloutBuffer:
    """Fixed [T, N] struct-of-tensors, allocated from a declarative schema.

    Linkage with src/protocol (the reason this class has no per-field code):

    * layout      -- BufferSchema fields: shape/dtype/extra_step come from the
                     declaration; nothing is hard-coded here.
    * write side  -- Field.write_op (STORE / ACCUMULATE / LAST) dispatches in
                     `_apply_write`; there is one write path.
    * sample side -- Field.sample_op (FLATTEN / WHOLE / SEQUENCE) + MinibatchSpec
                     + SampleMapping decide what `sample()` yields: a protocol
                     `RolloutSamples` NamedTuple (or the extended class built
                     from `extra_samples`), never an anonymous dict.
    * estimates   -- advantages / returns are estimator output slots, written via
                     `set_estimates` (or assigned directly, e.g. GA2E's per-epoch
                     rewrite); they are not schema fields and not writable by
                     `write()`.

    Static shapes everywhere: the precondition for compiling the update once.
    """

    def __init__(
        self,
        rollout_len: int,
        num_envs: int,
        schema: BufferSchema,
        device: str | torch.device = "cpu",
        spec: MinibatchSpec | None = None,
        mapping: SampleMapping | None = None,
        extra_samples: tuple[str, ...] = (),
    ):
        self.T = int(rollout_len)
        self.N = int(num_envs)
        self.device = torch.device(device)
        self.schema: BufferSchema = dict(schema)
        self._spec = spec or MinibatchSpec()
        self._mapping = mapping or SampleMapping()
        self._extra_samples = tuple(extra_samples)

        # -- declaration cross-check: a wrong name dies here, not mid-run ----- #
        validate_against_schema(self._mapping, self.schema, self._extra_samples)
        if self._spec.mode == "sequence":
            raise NotImplementedError(
                "sequence sampling is reserved for recurrent policies; use the "
                "flat mode for feed-forward algorithms"
            )
        for name in self._extra_samples:
            if self.schema[name].sample_op is SampleOp.SEQUENCE:
                raise ValueError(
                    f"extra sample field {name!r} has SampleOp.SEQUENCE, which the "
                    f"flat spec cannot cut; use sequence mode (recurrent) or WHOLE"
                )
        for element, src in self._mapping.sources().items():
            if element in ("advantages", "returns"):
                continue
            if src in self.schema and self.schema[src].sample_op is not SampleOp.FLATTEN:
                raise ValueError(
                    f"core sample element {element!r} maps to {src!r} with "
                    f"sample_op={self.schema[src].sample_op.value}; core elements "
                    f"are cut per sample and must be FLATTEN (declare WHOLE "
                    f"fields as extra_samples instead)"
                )

        # -- allocation: one tensor per field, static shapes ------------------ #
        self._buf: dict[str, Tensor] = {}
        for name, f in self.schema.items():
            steps = self.T + 1 if f.extra_step else self.T
            self._buf[name] = torch.zeros((steps, self.N, *f.shape),
                                          dtype=f.dtype, device=self.device)
        # estimator output slots
        self.advantages = torch.zeros((self.T, self.N), device=self.device)
        self.returns = torch.zeros((self.T, self.N), device=self.device)
        self.pos = 0
        self._samples_cls = (sample_class(self._extra_samples)
                             if self._extra_samples else RolloutSamples)

    # ------------------------------ write ------------------------------- #

    def reset(self) -> None:
        """Empty the rollout (pos -> 0). Tensors are left stale on purpose --
        every slot is rewritten during collection."""
        self.pos = 0

    def write(self, **kv) -> None:
        """Write fields for the current step WITHOUT advancing (a step is written
        in two parts: obs/action before env.step, reward/masks after). Unknown
        field names fail here, with the known list in the message."""
        if self.pos >= self.T:
            raise RuntimeError(f"buffer is full (T={self.T})")
        unknown = set(kv) - set(self.schema)
        if unknown:
            raise KeyError(
                f"undeclared fields {sorted(unknown)}; known: {sorted(self.schema)}"
            )
        for k, v in kv.items():
            self._apply_write(k, v)

    def write_obs(self, obs: "Tensor | dict[str, Tensor]") -> None:
        """Write the observation; dict obs expand to their obs.<key> fields."""
        if isinstance(obs, dict):
            for k, v in obs.items():
                self.write(**{f"obs.{k}": v})
        else:
            self.write(obs=obs)

    def write_masks(self, masks: Masks | Any) -> None:
        """Write the three masks from any object carrying them by attribute."""
        self.write(terminated=masks.terminated, truncated=masks.truncated,
                   valid=masks.valid)

    def advance(self) -> None:
        """The current step is fully written; move to the next slot."""
        self.pos += 1

    def _apply_write(self, k: str, v: Tensor) -> None:
        """Dispatch on Field.write_op -- the write side of the declaration."""
        op = self.schema[k].write_op
        val = self._coerce(k, v)
        if op is WriteOp.ACCUMULATE:
            self._buf[k][self.pos] += val
        else:  # STORE and LAST coincide for a single write per step
            self._buf[k][self.pos] = val

    def _coerce(self, k: str, v) -> Tensor:
        tgt = self._buf[k]
        want = tgt.shape[1:]                     # (N, *field.shape)
        t = torch.as_tensor(v, device=self.device)
        if tuple(t.shape) != want:
            # no silent reshape: a same-size, wrong-shape write must not pass
            raise ValueError(
                f"field {k!r} expects {want} for the current step, got "
                f"{tuple(t.shape)} (dtype {t.dtype}); declared field shape is "
                f"{self.schema[k].shape}"
            )
        return t.to(tgt.dtype)

    # ------------------------- estimator outputs ------------------------ #

    def set_bootstrap_value(self, v: Tensor) -> None:
        """Store V(s_T) into the value field's extra slot."""
        self._buf["value"][self.T] = torch.as_tensor(
            v, device=self.device).reshape(self.N).to(self._buf["value"].dtype)

    def set_estimates(self, advantages: Tensor, returns: Tensor) -> None:
        """Fill the estimator output slots. Per-epoch rewrites (GA2E) may assign
        `buf.advantages` directly instead."""
        for slot, val in ((self.advantages, advantages), (self.returns, returns)):
            t = torch.as_tensor(val, device=self.device)
            if tuple(t.shape) != (self.T, self.N):
                raise ValueError(f"estimate shape {tuple(t.shape)} != {(self.T, self.N)}")
            slot.copy_(t.detach().to(slot.dtype))

    # ------------------------------- read ------------------------------- #

    def __getitem__(self, k: str) -> Tensor:
        return self._buf[k]

    def __contains__(self, k: str) -> bool:
        return k in self._buf

    @property
    def values(self) -> Tensor:
        """[T, N] -- excludes the bootstrap slot."""
        return self._buf["value"][: self.T]

    @property
    def bootstrap_value(self) -> Tensor:
        return self._buf["value"][self.T]

    @property
    def obs(self) -> "Tensor | dict[str, Tensor]":
        if "obs" in self._buf:
            return self._buf["obs"][: self.T]
        return {k[4:]: v[: self.T] for k, v in self._buf.items()
                if k.startswith("obs.")}

    @property
    def masks(self) -> Masks:
        return Masks(terminated=self._buf["terminated"].bool(),
                     truncated=self._buf["truncated"].bool(),
                     valid=self._buf["valid"].bool())

    # ------------------------------ sample ------------------------------ #

    def sample(self, generator: torch.Generator | None = None
               ) -> Iterator[RolloutSamples]:
        """Yield minibatches: protocol-declared NamedTuples, invalid steps
        dropped, deterministic under `generator`."""
        if self._spec.mode != "flat":  # guarded again: spec may differ per call site
            raise NotImplementedError("sequence sampling is not implemented yet")
        valid = self._buf["valid"][: self.T].reshape(-1).bool()
        idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        perm = idx[torch.randperm(idx.numel(), generator=generator,
                                  device=idx.device)]
        n = max(1, perm.numel() // self._spec.num_minibatches)
        for start in range(0, perm.numel(), n):
            sel = perm[start:start + n]
            if sel.numel() == 0:
                continue
            yield self._assemble(sel)

    def _assemble(self, sel: Tensor) -> RolloutSamples:
        """One protocol sample: resolve every declared element from its source."""
        args: list[Any] = []
        for element, src in self._mapping.sources().items():
            if element == "observations":
                args.append(self._obs_at(sel))
            elif src == "advantages":
                args.append(self.advantages.reshape(-1)[sel])
            elif src == "returns":
                args.append(self.returns.reshape(-1)[sel])
            else:
                args.append(self._flat(src)[sel])
        for name in self._extra_samples:
            f = self.schema[name]
            if f.sample_op is SampleOp.WHOLE:
                args.append(self._buf[name])            # passed through uncut
            else:
                args.append(self._flat(name)[sel])
        return self._samples_cls(*args)

    def _flat(self, name: str) -> Tensor:
        return self._buf[name][: self.T].reshape(self.T * self.N,
                                                 *self.schema[name].shape)

    def _obs_at(self, sel: Tensor) -> "Tensor | dict[str, Tensor]":
        if "obs" in self._buf:
            return self._flat("obs")[sel]
        return {k[4:]: self._flat(k)[sel] for k in self._buf if k.startswith("obs.")}

    # ------------------------- time structure --------------------------- #

    def segments(self) -> list[tuple[int, int, int]]:
        """Contiguous trajectory segments as (env, start, end-exclusive).

        A segment ends at a termination, a truncation, an autoreset dummy step,
        or the rollout boundary. A view over the same storage -- estimators
        whose objective needs contiguous time index through these spans.
        """
        term = self._buf["terminated"][: self.T].bool()
        trunc = self._buf["truncated"][: self.T].bool()
        invalid = ~self._buf["valid"][: self.T].bool()
        out: list[tuple[int, int, int]] = []
        for n in range(self.N):
            start = None
            for t in range(self.T):
                if invalid[t, n]:
                    if start is not None:
                        out.append((n, start, t))
                        start = None
                    continue
                if start is None:
                    start = t
                if term[t, n] or trunc[t, n]:
                    out.append((n, start, t + 1))
                    start = None
            if start is not None:
                out.append((n, start, self.T))
        return out

    # ------------------------------ meta -------------------------------- #

    def describe(self) -> str:
        lines = [f"RolloutBuffer(T={self.T}, N={self.N}, device={self.device})"]
        lines.append(describe_schema(self.schema))
        lines.append("  advantages  [T, N] float32   # estimator output slot"
                     "\n  returns     [T, N] float32   # estimator output slot")
        lines.append(f"  samples     : {self._samples_cls.__name__}"
                     f"{self._extra_samples and self._extra_samples or ''}")
        lines.append(f"  spec        : {self._spec}")
        return "\n".join(lines)

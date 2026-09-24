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
    from protocol.masks import Masks
    from protocol.sample import (
        MinibatchSpec,
        RolloutSamples,
        SampleMapping,
        SampleOp,
        TrajectorySamples,
        sample_class,
        validate_against_schema,
    )
except ImportError:  # executed from inside src/memory/ without `src` on sys.path
    from protocol.buffer import BufferSchema, WriteOp, describe_schema  # type: ignore
    from protocol.masks import Masks  # type: ignore
    from protocol.sample import (  # type: ignore
        MinibatchSpec,
        RolloutSamples,
        SampleMapping,
        SampleOp,
        TrajectorySamples,
        sample_class,
        validate_against_schema,
    )


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
        if self._spec.mode == "flat":
            for name in self._extra_samples:
                if self.schema[name].sample_op is SampleOp.SEQUENCE:
                    raise ValueError(
                        f"extra sample field {name!r} has SampleOp.SEQUENCE, which "
                        f"the flat spec cannot cut; use WHOLE or mode='trajectory'"
                    )
            for element, src in self._mapping.sources().items():
                if element in ("advantages", "returns"):
                    continue
                if src in self.schema and self.schema[src].sample_op is not SampleOp.FLATTEN:
                    raise ValueError(
                        f"core sample element {element!r} maps to {src!r} with "
                        f"sample_op={self.schema[src].sample_op.value}; core "
                        f"elements are cut per sample in flat mode (declare WHOLE "
                        f"fields as extra_samples instead, or use "
                        f"mode='trajectory')"
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
        base_cls = TrajectorySamples if self._spec.mode == "trajectory" else RolloutSamples
        self._samples_cls = (sample_class(self._extra_samples, base=base_cls)
                             if self._extra_samples else base_cls)

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
        """Yield minibatches per the declared spec: flat mode yields shuffled
        per-sample batches; trajectory mode yields packed whole-trajectory
        batches. Deterministic under `generator`; invalid steps never appear."""
        if self._spec.mode == "flat":
            yield from self._sample_flat(generator)
        else:
            yield from self._sample_trajectories(generator)

    def _sample_flat(self, generator: torch.Generator | None) -> Iterator[Any]:
        valid = self._buf["valid"][: self.T].reshape(-1).bool()
        idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        perm = idx[torch.randperm(idx.numel(), generator=generator,
                                  device=idx.device)]
        # tensor_split: sizes differ by at most 1 -- a fixed chunk size would
        # dump the whole remainder into an orphan tail minibatch (possibly a
        # single sample, whose std() is NaN and once killed a whole policy)
        for sel in torch.tensor_split(perm, self._spec.num_minibatches):
            if sel.numel() > 0:
                yield self._assemble(sel)

    def _sample_trajectories(self, generator: torch.Generator | None) -> Iterator[Any]:
        """Pack whole trajectories up to `trajectory_frames` samples ("≈ m").

        Trajectories are shuffled, then drawn until the batch holds at least the
        target -- a trajectory is never cut, so a batch is ≥ target. The
        trailing leftover is discarded when it holds fewer than
        `min_trajectories` trajectories. Segments already exclude autoreset
        dummy steps, so drop_invalid is structural here.
        """
        segs = self.segments()
        if not segs:
            return
        total_frames = sum(e - s for _, s, e in segs)
        target = self._spec.trajectory_frames or max(
            1, total_frames // self._spec.num_minibatches)
        order = torch.randperm(len(segs), generator=generator,
                               device=self.device).tolist()
        batch: list[tuple[int, int, int]] = []
        frames = 0
        for i in order:
            n, s, e = segs[i]
            batch.append((n, s, e))
            frames += e - s
            if frames >= target:
                yield self._assemble_trajectory(batch)
                batch, frames = [], 0
        if batch and len(batch) >= self._spec.min_trajectories:
            yield self._assemble_trajectory(batch)

    def _assemble_trajectory(self, segs: list[tuple[int, int, int]]) -> Any:
        """One trajectory batch: elements concatenated along the sample axis in
        segment order, plus lengths and last_values (zeroed at true terminations).
        Assembled by name -- see _assemble."""
        kwargs: dict[str, Any] = {}
        for element, src in self._mapping.sources().items():
            if element == "observations":
                kwargs[element] = self._obs_concat(segs)
            elif src == "advantages":
                kwargs[element] = self._src_concat(self.advantages, segs)
            elif src == "returns":
                kwargs[element] = self._src_concat(self.returns, segs)
            else:
                kwargs[element] = self._src_concat(self._buf[src], segs)
        kwargs["lengths"] = torch.tensor([e - s for _, s, e in segs], dtype=torch.long,
                                         device=self.device)
        kwargs["last_values"] = self._last_values(segs)
        for name in self._extra_samples:
            f = self.schema[name]
            if f.sample_op is SampleOp.WHOLE:
                kwargs[name] = self._buf[name]
            else:  # FLATTEN and SEQUENCE both concatenate along whole trajectories
                kwargs[name] = self._src_concat(self._buf[name], segs)
        return self._samples_cls(**kwargs)

    def _src_concat(self, tensor: Tensor, segs: list[tuple[int, int, int]]) -> Tensor:
        return torch.cat([tensor[s:e, n] for n, s, e in segs], dim=0)

    def _obs_concat(self, segs: list[tuple[int, int, int]]) -> "Tensor | dict[str, Tensor]":
        if "obs" in self._buf:
            return self._src_concat(self._buf["obs"], segs)
        return {k[4:]: self._src_concat(self._buf[k], segs)
                for k in self._buf if k.startswith("obs.")}

    def _last_values(self, segs: list[tuple[int, int, int]]) -> Tensor:
        """V(s_end) per trajectory: the stored value one step past the segment's
        end -- the bootstrap slot when the segment runs to the rollout boundary;
        zero where the trajectory ended on a true termination."""
        vals = []
        term = self._buf["terminated"]
        for n, _, e in segs:
            v = self._buf["value"][e, n]
            if bool(term[e - 1, n]):
                v = torch.zeros_like(v)
            vals.append(v)
        return torch.stack(vals).to(torch.float32)

    def _assemble(self, sel: Tensor) -> RolloutSamples:
        """One protocol sample: resolve every declared element from its source.

        Elements are assembled **by name, never by position** -- the mapping's
        field order and the NamedTuple's field order are two different orders,
        and a positional zip silently swaps old_values/old_log_prob (a real bug
        this fixed; it surfaced as instant policy collapse with no error).
        """
        kwargs: dict[str, Any] = {}
        for element, src in self._mapping.sources().items():
            if element == "observations":
                kwargs[element] = self._obs_at(sel)
            elif src == "advantages":
                kwargs[element] = self.advantages.reshape(-1)[sel]
            elif src == "returns":
                kwargs[element] = self.returns.reshape(-1)[sel]
            else:
                kwargs[element] = self._flat(src)[sel]
        for name in self._extra_samples:
            f = self.schema[name]
            if f.sample_op is SampleOp.WHOLE:
                kwargs[name] = self._buf[name]          # passed through uncut
            else:
                kwargs[name] = self._flat(name)[sel]
        return self._samples_cls(**kwargs)

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

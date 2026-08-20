"""GPU-resident on-policy rollout buffer whose fields come from a declarative Schema.

Layout: struct-of-tensors -- one pre-allocated tensor [T, N, ...] per field, with
static shapes throughout (the precondition for `torch.compile` compiling once).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .schema import Op, Schema, base_schema
from .types import Masks, Obs


class Minibatch(dict):
    """A dict subclass that also supports attribute access, keeping some IDE support."""

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    @property
    def obs(self) -> Obs:
        """Reconstruct obs: for dict observations, gather all `obs.*` sub-fields."""
        if "obs" in self:
            return self["obs"]
        sub = {k[4:]: v for k, v in self.items() if k.startswith("obs.")}
        if not sub:
            raise AttributeError("obs")
        return sub


class TrajectoryBatch:
    """A batch of whole trajectories, concatenated flat.

    `lengths` lets a consumer recover per-trajectory views via `tensor.split(lengths)`.
    `last_values` holds V(s_end) per trajectory, already zeroed where the trajectory ended
    on a true termination (no future value to bootstrap from).
    """

    __slots__ = ("fields", "lengths", "last_values")

    def __init__(self, fields: dict[str, Tensor], lengths: list[int],
                 last_values: Tensor):
        self.fields = fields
        self.lengths = lengths
        self.last_values = last_values

    def __getitem__(self, k: str) -> Tensor:
        return self.fields[k]

    def __contains__(self, k: str) -> bool:
        return k in self.fields

    @property
    def obs(self) -> Obs:
        if "obs" in self.fields:
            return self.fields["obs"]
        return {k[4:]: v for k, v in self.fields.items() if k.startswith("obs.")}

    @property
    def n_frames(self) -> int:
        return sum(self.lengths)

    def split(self, key: str) -> tuple[Tensor, ...]:
        """Per-trajectory views of one field."""
        return self.fields[key].split(self.lengths)


class RolloutBuffer:
    """Fixed [T, N] layout. No variable-length episodes, no prioritized sampling and
    no CPU paging -- on-policy training does not need any of it."""

    def __init__(
        self,
        rollout_len: int,
        num_envs: int,
        obs_space,
        act_space,
        device: torch.device | str = "cpu",
        extra: Schema | None = None,
    ):
        self.T = int(rollout_len)
        self.N = int(num_envs)
        self.device = torch.device(device)

        self.schema: Schema = base_schema(obs_space, act_space)
        if extra:
            overlap = set(extra) & set(self.schema)
            if overlap:
                raise ValueError(f"extra fields collide with core fields: {sorted(overlap)}")
            self.schema.update(extra)

        self._buf: dict[str, Tensor] = {}
        for name, f in self.schema.items():
            steps = self.T + 1 if f.extra_step else self.T
            self._buf[name] = torch.zeros(
                (steps, self.N, *f.shape), dtype=f.dtype, device=self.device
            )

        # Produced by the advantage estimator.
        self.advantages = torch.zeros((self.T, self.N), device=self.device)
        self.returns = torch.zeros((self.T, self.N), device=self.device)
        self.pos = 0

    # ---------------- write ----------------

    def reset(self) -> None:
        self.pos = 0

    def write(self, **kv) -> None:
        """Write fields for the current step **without advancing `pos`**.

        A step is written in two parts (obs/action before `env.step`, reward/masks
        after), so writing and advancing must be separate. A misspelled field name
        fails here rather than three hours into training.
        """
        if self.pos >= self.T:
            raise RuntimeError(f"buffer is full (T={self.T})")
        unknown = set(kv) - set(self.schema)
        if unknown:
            raise KeyError(
                f"undeclared fields {sorted(unknown)}; known fields: {sorted(self.schema)}"
            )
        for k, v in kv.items():
            self._apply_write(k, v)

    def _apply_write(self, k: str, v) -> None:
        """Dispatch on `Field.write_op` -- the write side of the declarative operators."""
        op = self.schema[k].write_op
        val = self._coerce(k, v)
        if op is Op.ACCUMULATE:
            self._buf[k][self.pos] += val
        else:  # STORE and LAST coincide for a single write per step.
            self._buf[k][self.pos] = val

    def write_obs(self, obs: Obs) -> None:
        """Write the observation (dict obs are expanded automatically). Does not advance."""
        if isinstance(obs, dict):
            for k, v in obs.items():
                self._apply_write(f"obs.{k}", v)
        else:
            self._apply_write("obs", obs)

    def write_masks(self, m: Masks) -> None:
        """Does not advance `pos`."""
        for name in ("terminated", "truncated", "valid"):
            self._apply_write(name, getattr(m, name))

    def advance(self) -> None:
        """Current step is fully written; move to the next slot."""
        self.pos += 1

    def set_bootstrap_value(self, v: Tensor) -> None:
        """Store V(s_T) for the advantage estimator."""
        self._buf["value"][self.T] = v.reshape(self.N).to(self._buf["value"].dtype)

    def _coerce(self, k: str, v: Tensor) -> Tensor:
        tgt = self._buf[k]
        want = tgt.shape[1:]
        v = torch.as_tensor(v, device=self.device)
        if v.shape != want:
            v = v.reshape(want)
        return v.to(tgt.dtype)

    # ---------------- read ----------------

    def __getitem__(self, k: str) -> Tensor:
        return self._buf[k]

    def __contains__(self, k: str) -> bool:
        return k in self._buf

    @property
    def obs(self) -> Obs:
        if "obs" in self._buf:
            return self._buf["obs"][: self.T]
        return {
            k[4:]: v[: self.T] for k, v in self._buf.items() if k.startswith("obs.")
        }

    @property
    def masks(self) -> Masks:
        return Masks(
            terminated=self._buf["terminated"].bool(),
            truncated=self._buf["truncated"].bool(),
            valid=self._buf["valid"].bool(),
        )

    @property
    def values(self) -> Tensor:
        """[T, N] -- excludes the bootstrap slot."""
        return self._buf["value"][: self.T]

    @property
    def bootstrap_value(self) -> Tensor:
        return self._buf["value"][self.T]

    # ---------------- minibatch ----------------

    def iter_minibatches(self, num_minibatches: int, generator=None):
        """Flatten [T, N] and shuffle into minibatches (FlatSampler semantics).

        Only `valid=True` samples are yielded -- autoreset dummy steps are dropped
        here, so the algorithm layer never needs to know autoreset exists.
        """
        flat_valid = self._buf["valid"][: self.T].reshape(-1).bool()
        idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
        perm = idx[torch.randperm(idx.numel(), generator=generator, device=idx.device)]
        n = max(1, perm.numel() // num_minibatches)
        for start in range(0, perm.numel(), n):
            sel = perm[start : start + n]
            if sel.numel() == 0:
                continue
            yield self._gather(sel)

    def _gather(self, sel: Tensor) -> Minibatch:
        mb = Minibatch()
        for name, f in self.schema.items():
            # Dispatch on Field.sample_op -- the sample side of the operators.
            if f.sample_op is Op.WHOLE:
                mb[name] = self._buf[name]
            else:
                flat = self._buf[name][: self.T].reshape(self.T * self.N, *f.shape)
                mb[name] = flat[sel]
        mb["advantages"] = self.advantages.reshape(-1)[sel]
        mb["returns"] = self.returns.reshape(-1)[sel]
        # Flat indices, needed by e.g. V-MPO to slice the full pi_old distribution.
        mb["_flat_idx"] = sel
        return mb

    # ---------------- trajectory view ----------------

    def segments(self) -> list[tuple[int, int, int]]:
        """Split [T, N] into contiguous trajectory segments.

        Returns a list of (env_index, start_t, end_t) with end exclusive. A segment ends
        at a termination, a truncation, an autoreset dummy step, or the rollout boundary.

        This is a **view over the same storage**, not a second buffer. Estimators whose
        objective needs contiguous time (DAE's telescoping residual) index through these
        segments; nothing is copied or reallocated.
        """
        ended = (self._buf["terminated"][: self.T].bool()
                 | self._buf["truncated"][: self.T].bool())
        invalid = ~self._buf["valid"][: self.T].bool()
        out: list[tuple[int, int, int]] = []
        for n in range(self.N):
            start = None
            for t in range(self.T):
                if invalid[t, n]:
                    # A dummy step belongs to no trajectory: close any open segment.
                    if start is not None:
                        out.append((n, start, t))
                        start = None
                    continue
                if start is None:
                    start = t
                if ended[t, n]:
                    out.append((n, start, t + 1))
                    start = None
            if start is not None:
                out.append((n, start, self.T))
        return out

    def iter_trajectories(self, batch_frames: int | None = None, generator=None):
        """Yield batches of **whole trajectories**, shuffled.

        Unlike `iter_minibatches`, which flattens time away, this preserves contiguity
        within each trajectory -- the requirement for telescoping / n-step objectives.

        Args:
            batch_frames: approximate number of transitions per batch. None = one batch
                containing every trajectory.

        Yields:
            `TrajectoryBatch` with flat, concatenated tensors plus `lengths`, so a consumer
            can `split(lengths)` to recover per-trajectory views.
        """
        segs = self.segments()
        if not segs:
            return
        order = torch.randperm(len(segs), generator=generator).tolist()
        target = batch_frames or (self.T * self.N)

        batch: list[tuple[int, int, int]] = []
        frames = 0
        for i in order:
            seg = segs[i]
            batch.append(seg)
            frames += seg[2] - seg[1]
            if frames >= target:
                yield self._gather_trajs(batch)
                batch, frames = [], 0
        if batch:
            yield self._gather_trajs(batch)

    def _gather_trajs(self, segs: list[tuple[int, int, int]]) -> TrajectoryBatch:
        lengths = [e - s for _, s, e in segs]
        fields: dict[str, Tensor] = {}
        for name, f in self.schema.items():
            if f.sample_op is Op.WHOLE:
                fields[name] = self._buf[name]
                continue
            buf = self._buf[name]
            fields[name] = torch.cat([buf[s:e, n] for n, s, e in segs], dim=0)
        # V(s_end) for each trajectory: the stored value one step past its end, which is
        # the bootstrap slot when the trajectory runs to the rollout boundary.
        vbuf = self._buf["value"]
        last_values = torch.stack([vbuf[e, n] for n, s, e in segs])
        # A trajectory that ended on a real termination has no future value.
        term = self._buf["terminated"][: self.T].bool()
        ended = torch.tensor(
            [bool(term[e - 1, n]) for n, s, e in segs], device=self.device
        )
        last_values = torch.where(ended, torch.zeros_like(last_values), last_values)
        return TrajectoryBatch(fields=fields, lengths=lengths, last_values=last_values)

    def describe(self) -> str:
        """The schema is printable, so "what is in this buffer" is answerable."""
        lines = [f"RolloutBuffer(T={self.T}, N={self.N}, device={self.device})"]
        for name, f in sorted(self.schema.items()):
            steps = self.T + 1 if f.extra_step else self.T
            ops = "" if (f.write_op is Op.STORE and f.sample_op is Op.FLATTEN) \
                else f"  [{f.write_op.value}/{f.sample_op.value}]"
            doc = f"  # {f.doc}" if f.doc else ""
            lines.append(
                f"  {name:<12} [{steps}, {self.N}, {list(f.shape)}] "
                f"{str(f.dtype).replace('torch.', ''):<8}{ops}{doc}"
            )
        return "\n".join(lines)

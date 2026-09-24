# -*- encoding: utf-8 -*-
"""Aggregate a finished run into scalar metrics -- L1, dependency-free.

Reads the per-iteration ``metrics.jsonl`` the Logger writes and reduces each
seed's run to a few scalars, then reduces those across seeds to mean/std. Pure
stdlib (``json`` + ``statistics``) on purpose: L1 must not pull a plotting or
array dependency in -- the workflow report layer (L4) adds the optional plot on
top of these numbers.

Metric definitions (DESIGN_v2 report stage):
  * ``rew_last10`` / ``rew_last30`` -- mean of the last 10% / 30% of the training
    reward curve (the ``rollout/ep_rew_mean`` series, deduped by step).
  * ``final_eval`` -- the last recorded ``eval/ep_rew_mean`` (deterministic eval
    at the final iteration); NaN if the run logged no eval.
Across seeds we report population mean/std (the seeds ARE the whole set we ran,
not a sample; a single seed -> std 0.0), matching the unbiased=False convention
used elsewhere.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
#  reading one seed's metrics.jsonl
# --------------------------------------------------------------------------- #


def read_metrics(path: str | Path) -> list[dict]:
    """Parse a metrics.jsonl into a list of dump records.

    Blank lines and any corrupt/half-written line are skipped rather than
    raised -- a report must survive a crashed or in-progress run."""
    path = Path(path)
    records: list[dict] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


# Mirror of logger._CATEGORY, kept LOCAL so results.py stays stdlib-only (L1).
# The Logger groups each record into train/rollout/eval buckets; older runs are
# still flat. `_lookup` reads a flat, prefixed key from either layout.
_BUCKET = {
    "train/": "train", "loss/": "train", "grad/": "train",
    "diag/": "train", "time/": "train", "perf/": "train",
    "rollout/": "rollout", "charts/": "rollout",
    "eval/": "eval",
}


def _lookup(rec: dict, key: str):
    """Value for a flat prefixed metric key, tolerant of both record layouts.

    New grouped record: ``{"rollout": {"ep_rew_mean": ...}}``; legacy flat
    record: ``{"rollout/ep_rew_mean": ...}``. Returns None if absent. The
    grouping mirrors ``logger._CATEGORY``; a collision fallback that stored a
    one-level-prefixed leaf (``loss.policy``) is also honored."""
    if key in rec:                                  # legacy flat layout (fast path)
        return rec[key]
    for prefix, bucket in _BUCKET.items():
        if key.startswith(prefix):
            sub = rec.get(bucket)
            if isinstance(sub, dict):
                leaf = key[len(prefix):]
                if leaf in sub:
                    return sub[leaf]
                alt = key.replace("/", ".", 1)      # collision fallback name
                if alt in sub:
                    return sub[alt]
            return None
    return None


def _series(records: list[dict], key: str) -> list[tuple[int, float]]:
    """(step, value) points for `key`, deduped by step (last wins), step-ordered.

    The Logger re-emits the sliding-window metric on every dump, so the same
    step can appear more than once (e.g. the final flush repeats it); dedup by
    step keeps one point per step. Non-finite values are dropped. Reads either
    the grouped or the legacy-flat record layout via `_lookup`."""
    by_step: dict[int, float] = {}
    for r in records:
        if "step" not in r:
            continue
        raw = _lookup(r, key)
        if raw is None:
            continue
        try:
            v = float(raw)
            s = int(r["step"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            by_step[s] = v
    return [(s, by_step[s]) for s in sorted(by_step)]


def _tail_mean(values: list[float], frac: float) -> float:
    """Mean of the last `frac` of a series (at least one point)."""
    if not values:
        return float("nan")
    k = max(1, math.ceil(len(values) * frac))
    return statistics.fmean(values[-k:])


def _last_finite(records: list[dict], key: str) -> float:
    """The last finite value for `key` across records; NaN if never present.

    Used for `final_eval`: eval is only logged on eval iterations, and the very
    last dump line (the final flush) may carry only rollout metrics -- so we scan
    for the last line that actually HAS an eval value, not the last line. Reads
    either the grouped or the legacy-flat record layout via `_lookup`."""
    val = float("nan")
    for r in records:
        raw = _lookup(r, key)
        if raw is None:
            continue
        try:
            f = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            val = f
    return val


@dataclass
class SeedResult:
    """One seed's run reduced to scalars (+ its raw curves for plotting)."""

    seed: int
    rew_last10: float
    rew_last30: float
    final_eval: float
    curve: list[tuple[int, float]]      # (step, rollout/ep_rew_mean), training plot
    eval_curve: list[tuple[int, float]]  # (step, eval/ep_rew_mean), evaluation plot
    n_points: int


def seed_result(seed: int, metrics_path: str | Path) -> SeedResult:
    """Reduce one seed's metrics.jsonl to a `SeedResult`."""
    recs = read_metrics(metrics_path)
    curve = _series(recs, "rollout/ep_rew_mean")
    eval_curve = _series(recs, "eval/ep_rew_mean")
    ys = [v for _, v in curve]
    return SeedResult(
        seed=int(seed),
        rew_last10=_tail_mean(ys, 0.10),
        rew_last30=_tail_mean(ys, 0.30),
        final_eval=_last_finite(recs, "eval/ep_rew_mean"),
        curve=curve,
        eval_curve=eval_curve,
        n_points=len(curve),
    )


# --------------------------------------------------------------------------- #
#  reducing across seeds
# --------------------------------------------------------------------------- #


@dataclass
class Stat:
    """A scalar reduced across seeds: population mean/std over the finite values."""

    mean: float
    std: float
    n: int                     # how many seeds contributed a finite value
    values: list[float]        # per-seed values as given (may contain NaN)

    def as_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "n": self.n,
                "values": list(self.values)}


def _stat(values: list[float]) -> Stat:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return Stat(float("nan"), float("nan"), 0, list(values))
    mean = statistics.fmean(finite)
    std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    return Stat(mean, std, len(finite), list(values))


@dataclass
class GroupResult:
    """One (experiment, env) aggregated across its seeds."""

    label: str                 # the experiment slug (e.g. "epsilon=0.1")
    env: str
    seeds: list[SeedResult]
    rew_last10: Stat
    rew_last30: Stat
    final_eval: Stat
    overrides: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "experiment": self.label,
            "env": self.env,
            "overrides": dict(self.overrides),
            "n_seeds": len(self.seeds),
            "seeds": [{"seed": s.seed, "rew_last10": s.rew_last10,
                       "rew_last30": s.rew_last30, "final_eval": s.final_eval,
                       "n_points": s.n_points} for s in self.seeds],
            "final_eval": self.final_eval.as_dict(),
            "rew_last10": self.rew_last10.as_dict(),
            "rew_last30": self.rew_last30.as_dict(),
        }


def group_result(label: str, env: str, seed_results, overrides: dict | None = None
                 ) -> GroupResult:
    """Reduce a group's per-seed `SeedResult`s into cross-seed statistics."""
    sr = list(seed_results)
    return GroupResult(
        label=label, env=env, seeds=sr,
        rew_last10=_stat([s.rew_last10 for s in sr]),
        rew_last30=_stat([s.rew_last30 for s in sr]),
        final_eval=_stat([s.final_eval for s in sr]),
        overrides=dict(overrides or {}),
    )

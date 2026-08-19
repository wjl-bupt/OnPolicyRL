"""Logger -- SB3-compatible API (record / record_mean / dump) with inferred aggregation.

Three deliberate differences from SB3 (DESIGN.md §4.9):
  1. `record` picks its aggregation from the key prefix. In SB3 `record` overwrites by
     default, so using it where `record_mean` was needed silently drops data.
  2. `metrics.jsonl` is always written as the authoritative data source. SB3's CSV
     writer misaligns columns when the key set changes mid-run.
  3. The logger is passed explicitly rather than attached as `self.logger`, which makes
     running several experiments in one process safe.

Logging never raises into the training loop: a failing sink warns once and degrades.
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol


class Sink(Protocol):
    def write(self, metrics: dict[str, float], step: int) -> None: ...
    def write_media(self, key: str, value: Any, step: int) -> None: ...
    def close(self) -> None: ...


# Prefix -> aggregation. Unmatched keys (e.g. ga2e/*) default to mean, so an
# algorithm's own metrics need zero registration.
_AGG = {
    "train/": "mean", "loss/": "mean", "grad/": "mean",
    "diag/": "mean", "time/": "sum", "perf/": "last",
    "rollout/": "window", "charts/": "window",
}


def _agg_for(key: str) -> str:
    for pre, how in _AGG.items():
        if key.startswith(pre):
            return how
    return "mean"


class ConsoleSink:
    """SB3-style aligned table."""

    def __init__(self, stream=sys.stdout):
        self.stream = stream

    def write(self, metrics: dict[str, float], step: int) -> None:
        if not metrics:
            return
        w = max((len(k) for k in metrics), default=10)
        lines = [f"{'-' * (w + 16)}", f"| step {step:<{w + 8}} |"]
        for k in sorted(metrics):
            v = metrics[k]
            sv = f"{v:.4g}" if isinstance(v, float) else str(v)
            lines.append(f"| {k:<{w}} | {sv:>10} |")
        lines.append("-" * (w + 16))
        print("\n".join(lines), file=self.stream, flush=True)

    def write_media(self, key, value, step) -> None:
        pass

    def close(self) -> None:
        pass


class JsonlSink:
    """The authoritative data source: one line per dump, for offline analysis."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")

    def write(self, metrics: dict[str, float], step: int) -> None:
        rec = {"step": step, "wall_time": time.time(), **metrics}
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()

    def write_media(self, key, value, step) -> None:
        pass

    def close(self) -> None:
        self.f.close()


class TensorBoardSink:
    def __init__(self, log_dir: Path):
        from torch.utils.tensorboard import SummaryWriter  # deferred import

        self.w = SummaryWriter(str(log_dir))

    def write(self, metrics: dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self.w.add_scalar(k, v, step)

    def write_media(self, key, value, step) -> None:
        self.w.add_video(key, value, step)

    def close(self) -> None:
        self.w.close()


class Logger:
    def __init__(
        self,
        sinks: list[Sink] | None = None,
        run_dir: Path | str | None = None,
        window: int = 100,
    ):
        self.run_dir = Path(run_dir) if run_dir else None
        if sinks is None:
            sinks = [ConsoleSink()]
            if self.run_dir:
                sinks.append(JsonlSink(self.run_dir / "metrics.jsonl"))
        self.sinks = sinks
        self._acc: dict[str, list[float]] = {}
        self._how: dict[str, str] = {}
        self._win: dict[str, deque] = {}
        self.window = window
        self._failed: set[int] = set()
        self.ep_returns: deque = deque(maxlen=window)
        self.ep_lengths: deque = deque(maxlen=window)

    # ---------------- SB3-compatible API ----------------

    def record(self, key: str, value: Any) -> None:
        how = _agg_for(key)
        self._how[key] = how
        if how == "window":
            self._win.setdefault(key, deque(maxlen=self.window)).append(float(value))
        else:
            self._acc.setdefault(key, []).append(float(value))

    def record_mean(self, key: str, value: Any) -> None:
        self._how[key] = "mean"
        self._acc.setdefault(key, []).append(float(value))

    def dump(self, step: int) -> None:
        out: dict[str, float] = {}
        for k, vals in self._acc.items():
            how = self._how.get(k, "mean")
            if not vals:
                continue
            if how == "sum":
                out[k] = float(sum(vals))
            elif how == "last":
                out[k] = float(vals[-1])
            else:
                out[k] = float(sum(vals) / len(vals))
        for k, dq in self._win.items():
            if dq:
                out[k] = float(sum(dq) / len(dq))

        # time/* -> fractions, so "where the time goes" is visible at a glance.
        t_keys = [k for k in out if k.startswith("time/") and not k.endswith("_frac")]
        total = sum(out[k] for k in t_keys)
        if total > 0:
            for k in t_keys:
                out[f"{k}_frac"] = out[k] / total

        if self.ep_returns:
            out["rollout/ep_rew_mean"] = float(sum(self.ep_returns) / len(self.ep_returns))
            out["rollout/ep_len_mean"] = float(sum(self.ep_lengths) / len(self.ep_lengths))
            out["rollout/n_episodes"] = float(len(self.ep_returns))

        self._emit(out, step)
        self._acc.clear()

    # ---------------- convenience extensions ----------------

    def add(self, **kv) -> None:
        for k, v in kv.items():
            self.record(k, v)

    def add_episode(self, ret: float, length: int) -> None:
        """Un-normalized return -- must be recorded outside the RewardNormalizer."""
        self.ep_returns.append(float(ret))
        self.ep_lengths.append(int(length))

    def media(self, key: str, value: Any, step: int) -> None:
        self._safe(lambda s: s.write_media(key, value, step))

    def _emit(self, metrics: dict[str, float], step: int) -> None:
        self._safe(lambda s: s.write(metrics, step))

    def _safe(self, fn) -> None:
        """A failing sink must never stop training -- losing a six-hour run because a
        logging backend went away is unacceptable."""
        for i, s in enumerate(self.sinks):
            try:
                fn(s)
            except Exception as e:  # noqa: BLE001
                if i not in self._failed:
                    self._failed.add(i)
                    print(
                        f"[oprl] warning: sink {type(s).__name__} failed, "
                        f"degrading: {e}",
                        file=sys.stderr,
                    )

    def close(self) -> None:
        self._safe(lambda s: s.close())

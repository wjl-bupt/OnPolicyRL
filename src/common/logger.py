"""Colored message logging + SB3-compatible metric recording, in one object.

A training loop uses both concerns at the same time, so they share one class:

* **messages** -- debug / info / success / warning / error, ANSI-colored,
  level-filtered; warning & error go to stderr.
* **metrics**  -- record / dump with prefix-driven aggregation
  (doc/logger_design.md §4); `metrics.jsonl` is the authoritative data source.

Design decisions carried from doc/logger_design.md:

* `record` picks its aggregation from the key prefix -- SB3's `record` overwrites
  by default, so using it where `record_mean` was needed silently drops data.
* `metrics.jsonl` is written line-by-line with an immediate flush: killing the
  process never loses an already-dumped line.
* Anomalies go through `event()` -- console + `anomalies.jsonl`, **no stop
  semantics**: the decision about a bad run belongs to the analysis stage.
* Nothing here ever raises into the training loop. A failing sink is disabled
  after one warning; losing metrics is always better than losing a six-hour run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol

try:
    from common.metrics import format_table
except ImportError:  # executed from inside src/common/ without `src` on sys.path
    from metrics import format_table

# --------------------------------------------------------------------------- #
#  levels & ANSI colors
# --------------------------------------------------------------------------- #

DEBUG, INFO, SUCCESS, WARNING, ERROR = 10, 20, 25, 30, 40
_LEVELS = {"debug": DEBUG, "info": INFO, "success": SUCCESS,
           "warning": WARNING, "error": ERROR}
_TAG = {"debug": "DEBUG", "info": "INFO", "success": "SUCCESS",
        "warning": "WARN", "error": "ERROR"}
_CODE = {"gray": 90, "red": 31, "green": 32, "yellow": 33, "cyan": 36}
_LEVEL_COLOR = {"debug": "gray", "info": "cyan", "success": "green",
                "warning": "yellow", "error": "red"}
# warning/error go to stderr; the rest to stdout.
_STREAM = {"debug": sys.stdout, "info": sys.stdout, "success": sys.stdout,
           "warning": sys.stderr, "error": sys.stderr}


def supports_color(stream) -> bool:
    """True when the stream can take ANSI escapes: a TTY, not a dumb terminal,
    and neither NO_COLOR nor a missing FORCE_COLOR rule says otherwise."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return (hasattr(stream, "isatty") and stream.isatty()
            and os.environ.get("TERM") != "dumb")


def _colorize(text: str, color: str, enabled: bool, bold: bool = False) -> str:
    if not enabled or color not in _CODE:
        return text
    pre = f"\033[{_CODE[color]}m" + ("\033[1m" if bold else "")
    return f"{pre}{text}\033[0m"


# --------------------------------------------------------------------------- #
#  sinks
# --------------------------------------------------------------------------- #


class Sink(Protocol):
    def write(self, metrics: dict[str, float], step: int) -> None: ...
    def close(self) -> None: ...


class ConsoleSink:
    """Colored table, one per dump. `style` selects the layout -- see
    `metrics.format_table` (box / sb3 / tree / minimal)."""

    def __init__(self, stream=None, use_color: bool | None = None, style: str = "sb3"):
        self.stream = stream if stream is not None else sys.stdout
        self.color = supports_color(self.stream) if use_color is None else use_color
        self.style = style

    def write(self, metrics: dict[str, float], step: int) -> None:
        if not metrics:
            return
        table = format_table(metrics, step=step, style=self.style)
        if self.color:
            out = []
            for line in table.splitlines():
                # borders (unicode box or dashed) in cyan; content stays plain
                if line.startswith(("┌", "├", "└", "─", "-")):
                    out.append(_colorize(line, "cyan", True))
                else:
                    out.append(line)
            # emphasize the header row (the line right after the top border)
            if len(out) >= 2:
                out[1] = _colorize(out[1], "cyan", True, bold=True)
            table = "\n".join(out)
        print(table, file=self.stream, flush=True)

    def close(self) -> None:
        pass


class JsonlSink:
    """The authoritative data source: one JSON object per dump, flushed at once."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, metrics: dict[str, float], step: int) -> None:
        rec = {"step": step, "wall_time": time.time(), **metrics}
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# --------------------------------------------------------------------------- #
#  prefix -> aggregation (doc/logger_design.md §4)
# --------------------------------------------------------------------------- #

_AGG = {
    "train/": "mean", "loss/": "mean", "grad/": "mean", "diag/": "mean",
    "time/": "sum", "perf/": "last", "eval/": "last",
    "rollout/": "window", "charts/": "window",
}


def _agg_for(key: str) -> str:
    for prefix, how in _AGG.items():
        if key.startswith(prefix):
            return how
    return "mean"  # algorithm-specific keys (e.g. ga2e/*) need zero registration


# --------------------------------------------------------------------------- #
#  Logger
# --------------------------------------------------------------------------- #


class Logger:
    def __init__(
        self,
        run_dir: Path | str | None = None,
        sinks: list[Sink] | None = None,
        window: int = 100,
        level: str = "info",
        use_color: bool | None = None,
        table_style: str = "sb3",
    ):
        """`use_color=None` auto-detects per stream; `window` sizes both the
        rollout sliding mean and the `history()` buffer; `table_style` selects
        the console layout (sb3 / box / tree / minimal)."""
        self.run_dir = Path(run_dir) if run_dir is not None else None
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        if sinks is None:
            sinks = [ConsoleSink(use_color=use_color, style=table_style)]
            if self.run_dir is not None:
                sinks.append(JsonlSink(self.run_dir / "metrics.jsonl"))
        self.sinks: list[Sink] = list(sinks)

        self.window = int(window)
        self.min_level = _LEVELS.get(level, INFO)
        self.use_color = use_color

        self._acc: dict[str, list[float]] = {}          # pending, per dump
        self._how: dict[str, str] = {}                  # explicit overrides
        self._win: dict[str, deque] = {}                # window-aggregated keys
        self._hist: dict[str, deque] = {}               # post-dump history
        self._ep_ret: deque = deque(maxlen=self.window)  # raw, un-normalized
        self._ep_len: deque = deque(maxlen=self.window)
        self._anomalies = None                          # lazily opened file
        self._failed: set[int] = set()
        self._anomaly_failed = False

    # ----------------------------- messages ----------------------------- #

    def log(self, level: str, msg: str, **kv) -> None:
        """One colored message line: `[oprl][LEVEL] msg | k=v k=v`."""
        lv = _LEVELS.get(level, INFO)
        if lv < self.min_level:
            return
        stream = _STREAM.get(level, sys.stdout)
        on = supports_color(stream) if self.use_color is None else self.use_color
        tag = _colorize(f"{_TAG.get(level, level.upper()):<7}", _LEVEL_COLOR.get(level, "cyan"),
                        on, bold=lv >= WARNING)
        suffix = ""
        if kv:
            kv_txt = " ".join(f"{k}={v}" for k, v in kv.items())
            suffix = " | " + _colorize(kv_txt, "gray", on)
        try:
            print(f"[oprl][{tag}] {msg}{suffix}", file=stream, flush=True)
        except Exception:  # a closed stream must never kill training
            pass

    def debug(self, msg: str, **kv) -> None:
        self.log("debug", msg, **kv)

    def info(self, msg: str, **kv) -> None:
        self.log("info", msg, **kv)

    def success(self, msg: str, **kv) -> None:
        self.log("success", msg, **kv)

    def warning(self, msg: str, **kv) -> None:
        self.log("warning", msg, **kv)

    def error(self, msg: str, **kv) -> None:
        self.log("error", msg, **kv)

    # ------------------------- metrics: record -------------------------- #

    def record(self, key: str, value: Any) -> None:
        """Record one observation; aggregation follows the prefix table unless
        `record_mean` was used for this key earlier in the same dump cycle."""
        how = self._how.get(key) or _agg_for(key)
        self._how[key] = how
        v = float(value)
        if how == "window":
            self._win.setdefault(key, deque(maxlen=self.window)).append(v)
        else:
            self._acc.setdefault(key, []).append(v)

    def record_mean(self, key: str, value: Any) -> None:
        """SB3-compatible explicit mean (overrides the prefix table)."""
        self._how[key] = "mean"
        self._acc.setdefault(key, []).append(float(value))

    def add(self, **kv) -> None:
        for k, v in kv.items():
            self.record(k, v)

    def add_episode(self, ret: float, length: int) -> None:
        """Raw, un-normalized return -- must be recorded outside the reward
        normalizer, or the learning curve is meaningless."""
        self._ep_ret.append(float(ret))
        self._ep_len.append(int(length))

    # -------------------------- metrics: dump --------------------------- #

    def dump(self, step: int) -> None:
        """Aggregate pending values, emit to every sink, then archive into the
        history buffer. `time/*` also derives `time/*_frac` (sums to 1)."""
        out: dict[str, float] = {}
        for k, vals in self._acc.items():
            if not vals:
                continue
            how = self._how.get(k, "mean")
            if how == "sum":
                out[k] = float(sum(vals))
            elif how == "last":
                out[k] = float(vals[-1])
            else:
                out[k] = float(sum(vals) / len(vals))
        for k, dq in self._win.items():
            if dq:
                out[k] = float(sum(dq) / len(dq))

        t_keys = [k for k in out if k.startswith("time/") and not k.endswith("_frac")]
        total = sum(out[k] for k in t_keys)
        if total > 0:
            for k in t_keys:
                out[f"{k}_frac"] = out[k] / total

        if self._ep_ret:
            out["rollout/ep_rew_mean"] = float(sum(self._ep_ret) / len(self._ep_ret))
            out["rollout/ep_len_mean"] = float(sum(self._ep_len) / len(self._ep_len))
            out["rollout/n_episodes"] = float(len(self._ep_ret))

        if out:
            self._emit(out, step)
            for k, v in out.items():
                self._hist.setdefault(k, deque(maxlen=self.window)).append(v)
        self._acc.clear()

    def history(self, key: str, n: int | None = None) -> list[float]:
        """The last n dumped values of `key` (monitor's sliding-window source)."""
        dq = self._hist.get(key)
        if not dq:
            return []
        vals = list(dq)
        return vals if n is None else vals[-n:]

    # --------------------------- anomaly event --------------------------- #

    def event(self, kind: str, iteration: int | None = None,
              global_step: int | None = None, level: str = "warning", **payload) -> None:
        """An anomaly event: console line + `anomalies.jsonl`. **No stop
        semantics** -- what to do with a bad run is an analysis-stage decision."""
        rec: dict[str, Any] = {"wall_time": time.time(), "kind": kind,
                               "iteration": iteration, "global_step": global_step}
        rec.update(payload)
        if self.run_dir is not None:
            try:
                if self._anomalies is None:
                    self._anomalies = open(self.run_dir / "anomalies.jsonl", "a",
                                           encoding="utf-8")
                self._anomalies.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._anomalies.flush()
            except Exception as e:  # noqa: BLE001
                if not self._anomaly_failed:
                    self._anomaly_failed = True
                    print(f"[oprl] warning: cannot write anomalies.jsonl, "
                          f"events are console-only: {e}", file=sys.stderr)
        self.log(level, f"event:{kind}", **payload)

    # ------------------------------ internals ---------------------------- #

    def _emit(self, metrics: dict[str, float], step: int) -> None:
        for i, s in enumerate(self.sinks):
            try:
                s.write(metrics, step)
            except Exception as e:  # noqa: BLE001
                if i not in self._failed:
                    self._failed.add(i)
                    print(f"[oprl] warning: sink {type(s).__name__} failed, "
                          f"disabling it: {e}", file=sys.stderr)

    def close(self) -> None:
        for i, s in enumerate(self.sinks):
            try:
                s.close()
            except Exception as e:  # noqa: BLE001
                if i not in self._failed:
                    self._failed.add(i)
                    print(f"[oprl] warning: sink {type(s).__name__} failed on "
                          f"close: {e}", file=sys.stderr)
        if self._anomalies is not None:
            try:
                self._anomalies.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
#  self-check
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Visual + behavioural check: run `python src/common/logger.py`.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        log = Logger(run_dir=td)
        log.error("this is an error")
        log.warning("this is a warning", it=3)
        log.success("this is a success")
        log.info("this is info", steps=1024)
        log.debug("this debug line is filtered out at level=info")

        for i in range(5):
            log.record("loss/policy", -0.1 - i)   # mean-aggregated
        log.record("loss/policy", -0.14)          # ...plus one more observation
        log.record("diag/kl", 0.009)
        log.record("perf/sps", 1234.5)
        log.record("time/env", 0.7)
        log.record("time/fwd", 0.3)
        log.record("eval/ep_rew_mean", 187.3)
        log.add_episode(ret=199.5, length=200)
        log.dump(step=2048)

        log.event("kl_explode", iteration=96, global_step=2048, value=0.42,
                  threshold=0.2)
        assert len(log.history("diag/kl")) == 1
        assert abs(log.history("diag/kl")[0] - 0.009) < 1e-9
        log.dump(step=4096)   # window keys re-emitted, acc keys gone
        log.close()

        lines = [json.loads(l) for l in
                 open(Path(td) / "metrics.jsonl", encoding="utf-8")]
        assert lines[0]["step"] == 2048 and abs(lines[0]["diag/kl"] - 0.009) < 1e-9
        assert "loss/policy" not in lines[1], "acc keys must clear after dump"
        assert abs(lines[0]["time/env_frac"] + lines[0]["time/fwd_frac"] - 1.0) < 1e-9
        anom = [json.loads(l) for l in
                open(Path(td) / "anomalies.jsonl", encoding="utf-8")]
        assert anom[0]["kind"] == "kl_explode" and anom[0]["value"] == 0.42
        print("\nself-check passed")

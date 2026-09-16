"""Timing, critic diagnostics, and plain-text table rendering -- no I/O, no colors.

Pure stdlib + torch. The Logger (logger.py) owns all presentation (ANSI colors,
sinks); this module stays testable as plain functions.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any
import torch


class Timer:
    """Accumulates wall-clock time per label.

    Usage::

        timer = Timer()
        with timer("fwd"):
            ...
        timer.drain()   # -> {"time/fwd": 0.123, ...} and clears

    The fractions (`time/*_frac`) are derived by the Logger at dump time, so the
    "where does the time go" attribution is always normalized to 1 there.
    """

    def __init__(self) -> None:
        self.acc: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.acc[name] = self.acc.get(name, 0.0) + (time.perf_counter() - t0)

    def drain(self) -> dict[str, float]:
        """Emit accumulated labels as `time/<name>` seconds and clear."""
        out = {f"time/{k}": v for k, v in self.acc.items()}
        self.acc.clear()
        return out


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """1 - Var(y_true - y_pred) / Var(y_true).

    <= 0 means the critic explains none of the return variance -- i.e. it is not
    learning. An underrated diagnostic; the training loop should record it into
    `diag/explained_variance` every iteration.
    """
    var_y = y_true.var()
    if var_y == 0:
        return float("nan")
    return float(1.0 - (y_true - y_pred).var() / var_y)


# --------------------------------------------------------------------------- #
#  table styles
# --------------------------------------------------------------------------- #


def _fmt(v) -> str:
    """Compact value formatting: %.4g floats, plain ints/bools, nan/inf visible."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):  # nan / +-inf
            return repr(v)
        return f"{v:.4g}"
    return str(v)


def _group_items(metrics: dict) -> list[tuple[str, list[tuple[str, Any]]]]:
    """Split flat keys into prefix groups, keys sorted overall.

    "train/loss" -> group "train/", sub-key "loss". Keys without a slash go to
    the "" group, which renders first without a header.
    """
    groups: dict[str, list[tuple[str, Any]]] = {}
    for k in sorted(metrics):
        if "/" in k:
            g, sub = k.split("/", 1)
            groups.setdefault(g + "/", []).append((sub, metrics[k]))
        else:
            groups.setdefault("", []).append((k, metrics[k]))
    return sorted(groups.items(), key=lambda kv: (kv[0] != "", kv[0]))


def format_table(
    metrics: dict,
    step: int | None = None,
    title: str | None = None,
    style: str = "sb3",
) -> str:
    """Render a flat metric dict as a plain-text table (no ANSI escapes).

    The Logger's console sink wraps this in color; tests assert on it directly.

    Styles (choose via ConsoleSink(style=...)):

        "sb3"      replica of Stable-Baselines3's HumanOutputFormat: dashed
                   borders, one header row per `prefix/` group, sub-keys indented
                   by four spaces, values left-aligned after the pipe.
        "box"      unicode box, flat sorted keys, values right-aligned.
        "tree"     unicode box with the sb3 grouping (group header lines inside).
        "minimal"  no borders at all -- compact, for high-frequency dumps.
    """
    if style == "sb3":
        return _table_sb3(metrics, step, title)
    if style == "box":
        return _table_box(metrics, step, title)
    if style == "tree":
        return _table_tree(metrics, step, title)
    if style == "minimal":
        return _table_minimal(metrics, step, title)
    raise ValueError(f"unknown table style {style!r}; available: sb3, box, tree, minimal")


def _table_box(metrics: dict, step: int | None, title: str | None) -> str:
    items = sorted(metrics.items())
    w_key = max((len(k) for k, _ in items), default=6)
    w_val = max((len(_fmt(v)) for _, v in items), default=5)
    head = title if title is not None else (f"step {step}" if step is not None else "metrics")
    inner = max(w_key + w_val + 3, len(head) + 2)
    top, sep, bot = f"┌{'─' * inner}┐", f"├{'─' * inner}┤", f"└{'─' * inner}┘"
    lines = [top, f"│ {head:<{inner - 2}} │", sep]
    for k, v in items:
        lines.append(f"│ {k:<{w_key}} {_fmt(v):>{w_val}}{' ' * (inner - w_key - w_val - 3)} │")
    lines.append(bot)
    return "\n".join(lines)


def _table_sb3(metrics: dict, step: int | None, title: str | None) -> str:
    rows: list[tuple[str, str | None]] = []
    if title is not None:
        rows.append((title, ""))
    elif step is not None:
        rows.append(("step", str(step)))
    for g, items in _group_items(metrics):
        if g:
            rows.append((g, None))  # None = group header row
        for sub, v in items:
            rows.append(("    " + sub, _fmt(v)))
    w_key = max(len(k) for k, _ in rows)
    w_val = max((len(v) for _, v in rows if v is not None), default=5)
    dash = "-" * (w_key + w_val + 7)
    lines = [dash]
    for k, v in rows:
        if v is None:
            lines.append(dash)                       # border before each group
            lines.append(f"| {k:<{w_key}} |")
        else:
            lines.append(f"| {k:<{w_key}} | {v:<{w_val}} |")
    lines.append(dash)
    return "\n".join(lines)


def _table_tree(metrics: dict, step: int | None, title: str | None) -> str:
    head = title if title is not None else (f"step {step}" if step is not None else "metrics")
    return _tree_build(head, _group_items(metrics))


def _tree_build(head: str, groups) -> str:
    rows: list[tuple[str, str | None]] = [(head, None)]
    for g, items in groups:
        if g:
            rows.append((g, None))
        for sub, v in items:
            rows.append(("  " + sub, _fmt(v)))
    w_key = max(len(t) for t, _ in rows)
    w_val = max((len(v) for _, v in rows if v is not None), default=5)
    inner = max(w_key + w_val + 3, len(head) + 2)
    top, sep, bot = f"┌{'─' * inner}┐", f"├{'─' * inner}┤", f"└{'─' * inner}┘"
    lines = [top, f"│ {head:<{inner - 2}} │", sep]
    for text, v in rows[1:]:
        if v is None:
            lines.append(f"│ {text:<{inner - 2}} │")
        else:
            lines.append(f"│ {text:<{w_key}} {v:>{w_val}}{' ' * (inner - w_key - w_val - 3)} │")
    lines.append(bot)
    return "\n".join(lines)


def _table_minimal(metrics: dict, step: int | None, title: str | None) -> str:
    head = title if title is not None else (f"step {step}" if step is not None else "metrics")
    vals = [_fmt(v) for v in metrics.values()]
    w_val = max(map(len, vals), default=5) + 2
    lines = [head]
    for k, v in sorted(metrics.items()):
        lines.append(f"  {k:<28} {_fmt(v):>{w_val}}")
    return "\n".join(lines)

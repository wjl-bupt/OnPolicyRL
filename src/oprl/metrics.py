"""Timing attribution and diagnostic metrics -- makes "where time goes" visible."""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch


class Timer:
    """Surfaces where wall-clock time goes, so you do not optimize the wrong thing."""

    def __init__(self):
        self.acc: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.acc[name] = self.acc.get(name, 0.0) + (time.perf_counter() - t0)

    def drain(self) -> dict[str, float]:
        out = {f"time/{k}": v for k, v in self.acc.items()}
        self.acc.clear()
        return out


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """<= 0 means the critic is not learning at all -- an underrated diagnostic."""
    var_y = y_true.var()
    if var_y == 0:
        return float("nan")
    return float(1.0 - (y_true - y_pred).var() / var_y)

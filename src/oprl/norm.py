"""Observation / reward normalization.

**Normalizer state is part of the model and must be checkpointed** (DESIGN.md §4.10).
Omitting it means a restored policy sees a completely different input distribution --
a very common silent failure.
"""

from __future__ import annotations

import torch
from torch import Tensor


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...] = (), device="cpu", epsilon: float = 1e-4):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.var = torch.ones(shape, device=device, dtype=torch.float64)
        self.count = epsilon

    def update(self, x: Tensor) -> None:
        x = x.double()
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        n = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + n
        new_mean = self.mean + delta * n / tot
        m_a = self.var * self.count
        m_b = batch_var * n
        m2 = m_a + m_b + delta.pow(2) * self.count * n / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class ObsNormalizer:
    def __init__(self, shape, device="cpu", clip: float = 10.0):
        self.rms = RunningMeanStd(shape, device)
        self.clip = clip
        self.frozen = False

    def __call__(self, obs: Tensor) -> Tensor:
        flat = obs.reshape(-1, *self.rms.mean.shape)
        if not self.frozen:
            self.rms.update(flat)
        std = (self.rms.var + 1e-8).sqrt().float()
        return ((obs - self.rms.mean.float()) / std).clamp(-self.clip, self.clip)

    def freeze(self) -> None:
        """Apply statistics without updating them, so eval rollouts do not
        pollute training statistics."""
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    def state_dict(self) -> dict:
        return self.rms.state_dict()

    def load_state_dict(self, d: dict) -> None:
        self.rms.load_state_dict(d)


class RewardNormalizer:
    """Note: this normalizes the **variance of the discounted return**, not the raw
    reward -- which is why it needs `gamma`.

    Passing a gamma that differs from the algorithm's silently rescales advantages.
    """

    def __init__(self, num_envs: int, gamma: float = 0.99, device="cpu", clip: float = 10.0):
        self.rms = RunningMeanStd((), device)
        self.ret = torch.zeros(num_envs, device=device)
        self.gamma = gamma
        self.clip = clip
        self.frozen = False

    def __call__(self, reward: Tensor, terminated: Tensor, truncated: Tensor) -> Tensor:
        self.ret = self.ret * self.gamma + reward
        if not self.frozen:
            self.rms.update(self.ret.reshape(-1, 1).squeeze(-1).unsqueeze(-1).squeeze(-1).reshape(-1))
        std = (self.rms.var + 1e-8).sqrt().float()
        out = (reward / std).clamp(-self.clip, self.clip)
        done = terminated | truncated
        self.ret = torch.where(done, torch.zeros_like(self.ret), self.ret)
        return out

    def freeze(self) -> None:
        self.frozen = True

    def state_dict(self) -> dict:
        return {"rms": self.rms.state_dict(), "ret": self.ret}

    def load_state_dict(self, d: dict) -> None:
        self.rms.load_state_dict(d["rms"])
        self.ret = d["ret"]

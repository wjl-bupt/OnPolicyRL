"""Core types and protocols -- depends only on torch, never on other oprl modules.

The key design decision is in `Masks` (see DESIGN.md §4.1): the buffer never stores
a collapsed `done` flag, and no API returns one.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

import torch
from torch import Tensor

# Observation: a single tensor, or a dict (MiniGrid, goal-conditioned tasks).
Obs = Tensor | dict[str, Tensor]
# Recurrent state: (h, c) for LSTM, h for GRU, None for feed-forward policies.
RState = Tensor | tuple[Tensor, ...] | None


class Masks(NamedTuple):
    """Three masks with three distinct meanings -- the classic on-policy pitfall.

    terminated: true MDP termination; cuts the value bootstrap.
    truncated:  time-limit / external cutoff; the bootstrap is **kept** (the state
                still had value).
    valid:      False marks a dummy step produced by Gymnasium next-step autoreset;
                such steps must be excluded from the loss.

    The two masks play different roles in GAE::

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)
        adv_t   = delta_t + gamma * lam * (1 - (terminated_t | truncated_t)) * adv_{t+1}

    Collapsing them into a single `done` is the most widespread on-policy bug -- the
    reference CleanRL PPO does exactly that.
    """

    terminated: Tensor  # [T, N] bool
    truncated: Tensor  # [T, N] bool
    valid: Tensor  # [T, N] bool


@runtime_checkable
class EnvAdapter(Protocol):
    """Environment adapter.

    Contract: everything in and out is a `torch.Tensor` already living on
    `self.device`. That lets GPU-native environments (Isaac Lab, Brax, MJX) pass
    through with zero copies instead of being forced through a
    torch -> numpy -> torch round trip (DESIGN.md §4.2).
    """

    num_envs: int
    device: torch.device

    @property
    def obs_space(self) -> Any:
        """Observation space with single-env semantics."""
        ...

    @property
    def action_space(self) -> Any:
        """Action space with single-env semantics."""
        ...

    def reset(self, seed: int | None = None) -> Obs: ...

    def step(self, action: Tensor) -> tuple[Obs, Tensor, Masks, dict]:
        """Return (obs, reward, masks, info). `obs` is the **next** observation.

        Implementations must guarantee that the reward and masks of a terminal step
        describe the real final transition, and that `masks.valid` already reflects
        the adapter's own autoreset semantics.
        """
        ...

    def close(self) -> None: ...


@runtime_checkable
class Policy(Protocol):
    """The policy protocol -- the main interface between framework and user (§4.5).

    Only four methods. The framework does not care whether the internals are an MLP,
    a CNN, a Transformer or a two-tower model. Any `nn.Module` satisfying this
    protocol can be handed to `train()` directly, **without inheriting anything**.
    """

    is_recurrent: bool

    def act(
        self, obs: Obs, state: RState = None
    ) -> tuple[Tensor, Tensor, Tensor, RState]:
        """Sample during rollout. Returns (action, logprob, value, next_state)."""
        ...

    def evaluate(
        self,
        obs: Obs,
        action: Tensor,
        state: RState = None,
        valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Re-evaluate stored actions during the update. Returns (logprob, entropy, value).

        `valid` lets recurrent implementations mask sequences correctly; feed-forward
        implementations can ignore it.
        """
        ...

    def value(self, obs: Obs, state: RState = None) -> Tensor: ...

    def initial_state(self, batch: int) -> RState: ...

# -*- encoding: utf-8 -*-
'''
@File :advantages.py
@Created-Time :2026-09-16 17:58:59
@Author  :june
@Description   : Advantage Estimate
@Modified-Time :2026-09-16 17:58:59
'''
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch as th
from torch import Tensor


class AdvantageEstimator(str, Enum):
    """Pure-function advantage estimators (verl-style registry keys).

    Deliberately minimal, per verl's own note: the enum exists for spelling
    safety on the built-ins; a custom estimator just registers under a string.
    Stateful estimators (e.g. GA2E -- policy backprop + cross-iteration state)
    are estimator *components* (algo_design §5.1), not entries here.
    """

    GAE = "gae"


ADV_ESTIMATOR_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_adv_est(name: "str | AdvantageEstimator") -> Callable:
    """Register a pure-function advantage estimator under a name (verl contract).

    The function signature family is `(rewards, values, <bootstrap/masks>,
    gamma, gae_lambda) -> (advantages, returns)`; normalization does NOT happen
    here (see the GAE docstring).
    """

    def decorator(fn: Callable) -> Callable:
        key = name.value if isinstance(name, AdvantageEstimator) else str(name)
        prev = ADV_ESTIMATOR_REGISTRY.get(key)
        if prev is not None and prev is not fn:
            raise ValueError(
                f"adv estimator {key!r} is already registered: {prev!r} vs {fn!r}"
            )
        ADV_ESTIMATOR_REGISTRY[key] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name: "str | AdvantageEstimator") -> Callable:
    """Resolve an estimator by name (config strings land here)."""
    key = name.value if isinstance(name, AdvantageEstimator) else str(name)
    if key not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(
            f"Unknown advantage estimator: {key!r}; "
            f"available: {sorted(ADV_ESTIMATOR_REGISTRY)}"
        )
    return ADV_ESTIMATOR_REGISTRY[key]


# --------------------------------------------------------------------------- #
#  GAE (Schulman et al., 2016) -- contract below, implementation to fill in.
#
#  Reference: verl trainer/ppo/core_algos.py::compute_gae_advantage_return --
#  same name and (advantages, returns) convention. Two deliberate departures,
#  both because verl targets LLM RLHF and we target control tasks:
#
#  1. verl's single `eos_mask` plays BOTH roles (cut the bootstrap AND cut the
#     recursion). Here the two masks are separate arguments with different jobs
#     -- merging them is the most widespread on-policy bug (DESIGN.md §4.1):
#     a truncated state still has value, so truncation must KEEP the bootstrap.
#  2. verl pads the end of the sequence with 0.0 as next_values. Our buffer
#     keeps a real bootstrap V(s_T) in the value field's extra slot, so the
#     final step bootstraps from it instead of from zero.
# --------------------------------------------------------------------------- #


@register_adv_est(AdvantageEstimator.GAE)
@th.no_grad()
def compute_gae_advantage_return(
    rewards: Tensor,           # [T, N] float32 -- possibly normalized
    values: Tensor,            # [T, N] float32 -- V(s_t)
    last_value: Tensor,   # [N]    float32 -- V(s_T) from the buffer's extra slot
    terminated: Tensor,        # [T, N] bool    -- true MDP termination
    truncated: Tensor,         # [T, N] bool    -- time-limit / external cutoff
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Generalized Advantage Estimation over a [T, N] rollout.

    Contract (the implementation is held to every line of this docstring):

    Returns
    -------
    (advantages, returns), both [T, N] float32, detached.
    returns = advantages + values  (the critic's regression target).

    Semantics
    ---------
    Let next_values[t] = values[t+1] for t < T-1, and last_value[t] for
    t == T-1. The two masks do different jobs:

        delta_t = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
        adv_t   = delta_t + gamma * gae_lambda * (1 - (terminated[t] | truncated[t])) * adv_{t+1}

    with adv_T := 0. I.e.:

    * the value bootstrap is cut ONLY by `terminated` -- a truncated state still
      has value, so `truncated` must NOT zero next_values;
    * the advantage recursion is cut by EITHER flag;
    * adv_{t+1} beyond the rollout boundary is zero.

    Deliberately NOT an argument
    ----------------------------
    `valid` (the autoreset dummy-step mask) is excluded on purpose: dummy steps
    are dropped by the buffer's sample layer, and GAE must not branch on them
    here. Their advantage rows carry no meaning and are never consumed.
    (The LLM analog in verl is the response_mask carry-through inside its GAE
    loop -- "skip values and TD-error on observation tokens". Our structural
    equivalent already exists: a dummy step always follows a terminal step,
    where the recursion is cut anyway.)

    Deliberately NOT done here
    --------------------------
    Advantage whitening/normalization is NOT part of this function -- verl bakes
    `masked_whiten` into its GAE, but normalization is an update-stage policy
    (per-minibatch, configurable) in this framework and belongs to the loss
    assembly, not the estimator. This function returns raw advantages.

    Implementation notes (non-binding)
    ----------------------------------
    The backward scan (lastgaelam accumulation, verl-style) is the reference
    structure; a vectorized formulation is welcome if it passes the same tests.
    Do not mutate the input tensors.

    Acceptance tests the implementation must pass
    ---------------------------------------------
    1. element-wise equality against a brute-force per-trajectory loop;
    2. gae_lambda=0 collapses to one-step TD(0): adv_t == delta_t;
    3. gae_lambda=1 collapses to the discounted Monte-Carlo return minus baseline;
    4. truncation keeps the bootstrap, termination cuts it (the two-mask test
       in DESIGN.md §4.1 -- the bug this signature exists to prevent);
    5. shape/dtype: outputs [T, N] float32 regardless of input dtype.
    """
    rollout_len = values.shape[0]
    # contract item 5: float32 in, float32 out, regardless of the caller's dtype
    rewards = rewards.to(th.float32)
    values = values.to(th.float32)
    last_value = last_value.to(th.float32)
    # next_values[t] = values[t+1]; the last row bootstraps from V(s_T)
    next_values = th.cat([values[1:], last_value.unsqueeze(0)], dim=0)
    # delta: the bootstrap is cut ONLY by `terminated` (truncation keeps it)
    deltas = rewards + gamma * (1 - terminated.to(th.float32)) * next_values - values
    # recursion: cut by EITHER flag -- otherwise the next episode's advantage
    # leaks across a truncation boundary (the two-mask bug, DESIGN.md §4.1)
    cont = 1.0 - (terminated | truncated).to(th.float32)
    lastgaelam = th.zeros(values.shape[1], dtype=th.float32, device=values.device)
    gae = th.zeros_like(rewards)
    for t in reversed(range(0, rollout_len)):
        lastgaelam = deltas[t] + gamma * gae_lambda * cont[t] * lastgaelam
        gae[t] = lastgaelam

    returns = values + gae
    return gae, returns

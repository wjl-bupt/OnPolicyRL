"""Deterministic policy evaluation on a fresh env -- a primitive (L1).

Evaluation is separate from the training rollout on purpose:

* a **fresh, independent** env (built by the caller's ``env_factory``) so eval
  never disturbs the training env's autoreset/RNG stream;
* **RAW returns** read from ``info["_finished_episodes"]`` (the adapter surfaces
  un-normalized episode returns there) -- an eval curve must never see shaped or
  normalized rewards;
* **deterministic** actions by default (distribution mode: argmax for Categorical,
  mean for Normal). Deterministic eval draws nothing from the global RNG, so it
  does not perturb a training run's reproducibility -- and two eval calls with the
  same seed give identical scores.

Returns SB3-style ``eval/*`` metrics; the caller (Monitor) records them straight
into the logger, whose ``eval/`` prefix already aggregates as "last".
"""

from __future__ import annotations

from typing import Callable

import torch as th
from torch.distributions import Categorical, Normal


def _greedy_action(policy, obs) -> th.Tensor:
    """The distribution's mode -- argmax (Categorical) / mean (Normal). Falls
    back to a sample for any other distribution (best effort)."""
    if callable(getattr(policy, 'predict_action', None)):
        return policy.predict_action(obs, deterministic=True)
    d = policy.dist(obs)
    if isinstance(d, Categorical):
        return d.probs.argmax(dim=-1)
    if isinstance(d, Normal):
        return d.mean
    return d.sample()


@th.no_grad()
def evaluate(policy, env_factory: Callable[[], object], *, episodes: int = 10,
             seed: int = 0, deterministic: bool = True) -> dict[str, float]:
    """Run ``policy`` for ``episodes`` complete episodes on a fresh env.

    ``env_factory()`` must return a GymVectorAdapter-like object (``reset(seed)``,
    ``step(actions) -> (obs, reward, masks, info)`` with tensor I/O and
    ``info["_finished_episodes"]``). The env is built here and closed on exit.
    """
    env = env_factory()
    was_training = getattr(policy, "training", False)
    if hasattr(policy, "eval"):
        policy.eval()                              # no-op today; correct if BN/dropout appears
    returns: list[float] = []
    lengths: list[float] = []
    achievements = []
    # Fixed quotas avoid favoring short episodes in vectorized evaluation.
    n_envs = int(getattr(env, 'num_envs', 1))
    targets = [(episodes + i) // n_envs for i in range(n_envs)]
    counts = [0] * n_envs
    try:
        obs = env.reset(seed=seed)
        # generous safety cap: a runaway env must not hang the run forever
        for _ in range(max(1, episodes) * 10_000):
            action = _greedy_action(policy, obs) if deterministic \
                else policy.act(obs)[0]
            obs, _reward, _masks, info = env.step(action)
            finished = info.get("_finished_episodes", [])
            indices = info.get('_finished_episode_indices')
            if n_envs > 1 and indices is None:
                raise ValueError('Vector evaluation requires finished episode indices')
            completed_achievements = info.get("_finished_achievements", [])
            for position, (index, (ep_ret, ep_len)) in enumerate(zip(indices if indices is not None else [0] * len(finished), finished)):
                if counts[index] >= targets[index]:
                    continue
                counts[index] += 1
                returns.append(float(ep_ret))
                lengths.append(float(ep_len))
                if position < len(completed_achievements):
                    achievements.append(completed_achievements[position])
            if len(returns) >= episodes:
                break
    finally:
        if hasattr(env, "close"):
            env.close()
        if was_training and hasattr(policy, "train"):
            policy.train()

    returns, lengths = returns[:episodes], lengths[:episodes]
    n = len(returns)
    if n == 0:                                     # no episode finished in the cap
        return {"eval/ep_rew_mean": float("nan"), "eval/ep_rew_std": float("nan"),
                "eval/ep_len_mean": float("nan"), "eval/n_episodes": 0.0}
    rew = th.tensor(returns)
    length = th.tensor(lengths)
    result = {
        "eval/ep_rew_mean": float(rew.mean()),
        "eval/ep_rew_std": float(rew.std(unbiased=False)),   # 1-episode std = 0, not NaN
        "eval/ep_len_mean": float(length.mean()),
        "eval/n_episodes": float(n),
    }

    if achievements:
        rates = th.tensor(achievements, dtype=th.float32).gt(0).float().mean(0)
        result.update({f'eval/achievement_{i}': float(r) for i, r in enumerate(rates)})
        result['eval/achievements_mean'] = float(rates.sum())
    return result

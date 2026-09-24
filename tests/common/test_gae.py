"""GAE acceptance tests -- the five contract items from
`common/advantages.py::compute_gae_advantage_return`, plus the registry.

The brute-force reference is deliberately structured differently from the
implementation (per-trajectory segment splitting instead of a masked backward
scan over the whole [T, N] grid), so agreement is evidence, not tautology.
"""

import torch
import pytest

from common.advantages import (
    AdvantageEstimator,
    compute_gae_advantage_return,
    get_adv_estimator_fn,
)


def _rollout(seed: int, T: int = 16, N: int = 4,
             p_term: float = 0.12, p_trunc: float = 0.12):
    g = torch.Generator().manual_seed(seed)
    rewards = torch.randn(T, N, generator=g, dtype=torch.float64)
    values = torch.randn(T, N, generator=g, dtype=torch.float64)
    bootstrap = torch.randn(N, generator=g, dtype=torch.float64)
    terminated = torch.rand(T, N, generator=g) < p_term
    truncated = (torch.rand(T, N, generator=g) < p_trunc) & ~terminated
    return rewards, values, bootstrap, terminated, truncated


def _brute_force_gae(rewards, values, bootstrap, terminated, truncated, gamma, lam):
    """Segment-splitting reference: each column is split into trajectories at
    (terminated | truncated) boundaries; GAE runs backward inside each segment
    with the accumulator reset at the segment's last step."""
    T, N = rewards.shape
    adv = torch.zeros(T, N, dtype=torch.float64)
    for n in range(N):
        segs, start = [], 0
        for t in range(T):
            if bool(terminated[t, n] or truncated[t, n]):
                segs.append((start, t))
                start = t + 1
        if start < T:
            segs.append((start, T - 1))
        for s, e in segs:
            gae = 0.0
            for t in range(e, s - 1, -1):
                nv = float(bootstrap[n]) if t == T - 1 else float(values[t + 1, n])
                keep = 0.0 if bool(terminated[t, n]) else 1.0   # bootstrap cut by term only
                delta = float(rewards[t, n]) + gamma * nv * keep - float(values[t, n])
                gae = delta + gamma * lam * gae                  # reset at segment end
                adv[t, n] = gae
    return adv


def _call(rewards, values, bootstrap, terminated, truncated, gamma=0.99, lam=0.95):
    return compute_gae_advantage_return(rewards, values, bootstrap,
                                        terminated, truncated, gamma, lam)


# --------------------------------------------------------------------------- #
#  1. brute-force agreement (mixed terminations AND truncations)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force_reference(seed):
    args = _rollout(seed)
    adv, ret = _call(*args)
    expected = _brute_force_gae(*args, gamma=0.99, lam=0.95)
    assert torch.allclose(adv.double(), expected, atol=1e-5)
    assert torch.allclose(ret.double(), expected + args[1], atol=1e-5)


def test_lambda_zero_collapses_to_td0():
    rewards, values, bootstrap, terminated, truncated = _rollout(7)
    adv, _ = _call(rewards, values, bootstrap, terminated, truncated, lam=0.0)
    next_values = torch.cat([values[1:], bootstrap.unsqueeze(0)], dim=0)
    delta = rewards + 0.99 * (1 - terminated.double()) * next_values - values
    assert torch.allclose(adv.double(), delta, atol=1e-6)


def test_lambda_one_is_mc_minus_baseline_without_boundaries():
    """No terminations/truncations: adv_t = discounted reward-to-go
    + gamma^(T-t) * bootstrap - v_t, built independently of the recursion."""
    T, N, gamma = 12, 3, 0.99
    rewards, values, bootstrap, term, trunc = _rollout(3, T=T, N=N,
                                                       p_term=0.0, p_trunc=0.0)
    adv, ret = _call(rewards, values, bootstrap, term, trunc, gamma=gamma, lam=1.0)
    r2g = torch.zeros(T, N, dtype=torch.float64)
    acc = torch.zeros(N, dtype=torch.float64)
    for t in reversed(range(T)):
        acc = rewards[t] + gamma * acc
        r2g[t] = acc
    steps = torch.arange(T, 0, -1, dtype=torch.float64).unsqueeze(1)   # T..1
    expected = r2g + (gamma ** steps) * bootstrap.unsqueeze(0) - values
    assert torch.allclose(adv.double(), expected, atol=1e-6)


def test_truncation_keeps_bootstrap_termination_cuts_it():
    """The two-mask test (DESIGN.md §4.1): same rollout shape, env 0 truncated
    at t=1, env 1 terminated at t=1 -- the bootstrap must differ."""
    T, N, gamma, lam = 4, 2, 0.5, 0.8
    rewards = torch.ones(T, N, dtype=torch.float64)
    values = torch.arange(T, dtype=torch.float64).unsqueeze(1).expand(T, N) * 0.1
    bootstrap = torch.full((N,), 2.0, dtype=torch.float64)
    terminated = torch.zeros(T, N, dtype=torch.bool)
    truncated = torch.zeros(T, N, dtype=torch.bool)
    truncated[1, 0] = True
    terminated[1, 1] = True

    adv, ret = _call(rewards, values, bootstrap, terminated, truncated,
                     gamma=gamma, lam=lam)

    # env 0, t=1 truncated: bootstrap KEPT (values[2]) but recursion cut (no
    # adv[2] leaks in), and the continuation value stays out of later steps
    assert torch.allclose(adv[1, 0].double(),
                          torch.tensor(1.0 + gamma * values[2, 0] - values[1, 0]))
    # env 1, t=1 terminated: bootstrap CUT
    assert torch.allclose(adv[1, 1].double(),
                          torch.tensor(1.0 - values[1, 1]))
    # the two columns' boundary advantages differ -- the masks did different jobs
    assert not torch.allclose(adv[1, 0], adv[1, 1])
    # recursion crosses the boundary nowhere: adv[0] = delta[0] + gamma*lam*adv[1]
    for n in range(N):
        d0 = 1.0 + gamma * values[1, n] - values[0, n]
        assert torch.allclose(adv[0, n].double(), d0 + gamma * lam * adv[1, n].double())


def test_output_is_float32_and_inputs_are_not_mutated():
    rewards, values, bootstrap, terminated, truncated = _rollout(9)
    keep = [t.clone() for t in (rewards, values, bootstrap, terminated, truncated)]
    adv, ret = _call(rewards, values, bootstrap, terminated, truncated)
    assert adv.dtype == torch.float32 and ret.dtype == torch.float32
    for t, k in zip((rewards, values, bootstrap, terminated, truncated), keep):
        assert torch.equal(t, k)


def test_registry_resolves_gae_and_rejects_unknown():
    fn = get_adv_estimator_fn("gae")
    assert fn is compute_gae_advantage_return
    assert get_adv_estimator_fn(AdvantageEstimator.GAE) is fn
    with pytest.raises(ValueError, match="available"):
        get_adv_estimator_fn("no_such_estimator")

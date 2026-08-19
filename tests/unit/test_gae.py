"""GAE correctness tests -- the framework's most important set of assertions.

`test_truncation_preserves_bootstrap` pins down the industry-wide bug of collapsing
`terminated` and `truncated` into one `done`, which the reference CleanRL PPO does.
"""

import torch

from oprl import Masks, gae


def _brute_gae(rewards, values, boot, term, trunc, gamma, lam):
    """Brute-force reference: an element-wise Python loop cross-checking the vectorized one."""
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    for n in range(N):
        last = 0.0
        for t in range(T - 1, -1, -1):
            v_next = values[t + 1, n] if t + 1 < T else boot[n]
            # The bootstrap is cut only by `terminated`.
            nt = 0.0 if bool(term[t, n]) else 1.0
            delta = rewards[t, n] + gamma * v_next * nt - values[t, n]
            # The recursion is cut by either flag.
            cont = 0.0 if (bool(term[t, n]) or bool(trunc[t, n])) else 1.0
            last = delta + gamma * lam * cont * last
            adv[t, n] = last
    return adv


def _rand(T=8, N=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(T, N, generator=g),
        torch.randn(T, N, generator=g),
        torch.randn(N, generator=g),
    )


def test_gae_matches_brute_force():
    rew, val, boot = _rand()
    term = torch.zeros(8, 3, dtype=torch.bool)
    trunc = torch.zeros(8, 3, dtype=torch.bool)
    term[5, 0] = True
    trunc[3, 1] = True
    trunc[6, 2] = True

    adv, ret = gae(rew, val, boot, Masks(term, trunc, torch.ones_like(term)), 0.99, 0.95)
    expect = _brute_gae(rew, val, boot, term, trunc, 0.99, 0.95)
    assert torch.allclose(adv, expect, atol=1e-5)
    assert torch.allclose(ret, adv + val, atol=1e-6)


def test_lambda_zero_is_td0():
    """With lam=0, GAE reduces to the TD(0) residual."""
    rew, val, boot = _rand(seed=1)
    m = Masks(*[torch.zeros(8, 3, dtype=torch.bool)] * 2, torch.ones(8, 3, dtype=torch.bool))
    adv, _ = gae(rew, val, boot, m, 0.9, 0.0)
    nxt = torch.cat([val[1:], boot.reshape(1, -1)], 0)
    assert torch.allclose(adv, rew + 0.9 * nxt - val, atol=1e-6)


def test_lambda_one_is_monte_carlo():
    """With lam=1, GAE reduces to the Monte-Carlo return minus the baseline."""
    T, N, gamma = 6, 2, 0.9
    rew = torch.ones(T, N)
    val = torch.zeros(T, N)
    boot = torch.zeros(N)
    m = Masks(*[torch.zeros(T, N, dtype=torch.bool)] * 2, torch.ones(T, N, dtype=torch.bool))
    adv, _ = gae(rew, val, boot, m, gamma, 1.0)
    for t in range(T):
        mc = sum(gamma**i for i in range(T - t))
        assert abs(adv[t, 0].item() - mc) < 1e-5


def test_truncation_preserves_bootstrap():
    """**The core test**: truncation must keep the bootstrap, termination must cut it.

    Setup: a one-step rollout with reward=0 and V(s_next)=10.
      - truncated  -> delta = 0 + gamma*10 - V(s) = 9.9 - 1 = 8.9
      - terminated -> delta = 0 + 0        - V(s) = -1
    An implementation that collapses both into `done` produces identical results, and
    this test fails immediately.
    """
    rew = torch.zeros(1, 1)
    val = torch.ones(1, 1)
    boot = torch.full((1,), 10.0)
    ones = torch.ones(1, 1, dtype=torch.bool)
    zeros = torch.zeros(1, 1, dtype=torch.bool)

    adv_tr, _ = gae(rew, val, boot, Masks(zeros, ones, ones), 0.99, 0.95)
    adv_te, _ = gae(rew, val, boot, Masks(ones, zeros, ones), 0.99, 0.95)

    assert abs(adv_tr.item() - (0.99 * 10.0 - 1.0)) < 1e-5, "truncated must keep bootstrap"
    assert abs(adv_te.item() - (-1.0)) < 1e-5, "terminated must cut bootstrap"
    assert adv_tr.item() != adv_te.item(), "the two masks were collapsed into `done`"


def test_infinite_horizon_value_converges():
    """Analytic test: with no true termination, constant reward 1 and only a time limit,
    V should equal 1/(1-gamma).

    A correct implementation (bootstrap kept) yields advantages near zero, because V is
    already the exact value. A wrong one (collapsed `done`) produces a large negative
    advantage at the truncation point.
    """
    T, N, gamma = 20, 1, 0.9
    v_star = 1.0 / (1.0 - gamma)  # = 10
    rew = torch.ones(T, N)
    val = torch.full((T, N), v_star)
    boot = torch.full((N,), v_star)
    term = torch.zeros(T, N, dtype=torch.bool)
    trunc = torch.zeros(T, N, dtype=torch.bool)
    trunc[-1, 0] = True  # time limit only, not a real termination

    adv, _ = gae(rew, val, boot, Masks(term, trunc, torch.ones_like(term)), gamma, 0.95)
    assert adv.abs().max().item() < 1e-4, (
        f"advantage should be 0 when V=1/(1-gamma), got {adv.abs().max().item()}"
    )

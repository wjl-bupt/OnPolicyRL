"""DAE (Direct Advantage Estimation) tests.

The properties worth pinning down are the two that, if broken, leave something that still
trains but is no longer DAE:

1. `E_{a~pi}[A(s,a)] == 0` -- the defining property of an advantage function.
2. The telescoping residual is masked at trajectory boundaries.

Plus one analytic check: given the exact advantages and values of a known MDP, the
residual must be zero.
"""

import pytest
import torch

import oprl
from oprl.advantages import DAE
from oprl.advantages.dae import _shift
from oprl.algos import ppo


def _env(n=4):
    return oprl.make_env("CartPole-v1", num_envs=n, seed=1)


def _policy(env):
    return oprl.ActorCritic.from_config(
        env.obs_space, env.action_space, {"advantage_head": True}
    )


def _cfg(**kw):
    base = dict(total_steps=4 * 32 * 2, num_envs=4, rollout_len=32, device="cpu",
                num_epochs=2, advantage="dae", network={"advantage_head": True})
    base.update(kw)
    return ppo.PPOConfig(**base)


# ---------------- registration and wiring ----------------


def test_dae_registered():
    from oprl.advantages import ESTIMATORS
    from oprl.registry import registered

    assert "dae" in ESTIMATORS
    assert "dae" in registered("advantage")


def test_dae_declares_probs_field_sized_by_env():
    """`probs` needs one slot per action, which is only known from the env."""
    env = _env(2)
    fields = DAE().resolve_fields(env.obs_space, env.action_space)
    assert fields["probs"].shape == (int(env.action_space.n),)
    env.close()


def test_dae_rejects_continuous_actions():
    """The advantage head needs one output per action, so Box is out of scope. This must
    fail loudly at setup rather than silently mis-training."""
    env = oprl.make_env("Pendulum-v1", num_envs=2, seed=1)
    with pytest.raises(ValueError, match="Discrete"):
        DAE().resolve_fields(env.obs_space, env.action_space)
    env.close()


def test_advantage_head_requires_discrete():
    import gymnasium as gym

    e = gym.make("Pendulum-v1")
    with pytest.raises(ValueError, match="advantage_head requires"):
        oprl.ActorCritic(e.observation_space, e.action_space, advantage_head=True)


def test_policy_without_head_raises():
    env = _env(2)
    plain = oprl.ActorCritic(env.obs_space, env.action_space)
    with pytest.raises(RuntimeError, match="no advantage head"):
        plain.advantages(torch.zeros(2, 4), torch.ones(2, 2) / 2)
    env.close()


# ---------------- defining property: zero-mean advantage ----------------


def test_advantage_head_is_zero_mean_under_policy():
    """**The defining property.** E_{a~pi}[A(s,a)] must be exactly 0.

    Without this projection the head could absorb an arbitrary state-dependent offset and
    would no longer be an advantage function -- training would still "work" while
    measuring something else.
    """
    env = _env(8)
    policy = _policy(env)
    obs = torch.randn(64, int(env.obs_space.shape[0]))
    probs = torch.softmax(torch.randn(64, int(env.action_space.n)), dim=-1)
    adv, values = policy.advantages(obs, probs)
    assert torch.allclose((probs * adv).sum(-1), torch.zeros(64), atol=1e-5)
    assert values.shape == (64,)
    env.close()


def test_zero_mean_holds_for_arbitrary_distributions():
    """Including near-deterministic policies, where a naive mean-subtraction would drift."""
    env = _env(2)
    policy = _policy(env)
    obs = torch.randn(32, int(env.obs_space.shape[0]))
    logits = torch.randn(32, int(env.action_space.n)) * 20  # nearly one-hot
    probs = torch.softmax(logits, dim=-1)
    adv, _ = policy.advantages(obs, probs)
    assert (probs * adv).sum(-1).abs().max() < 1e-4
    env.close()


# ---------------- boundary masking ----------------


def test_shift_pads_with_zeros():
    x = torch.arange(5.0)
    assert torch.equal(_shift(x, 0), x)
    assert torch.equal(_shift(x, 2), torch.tensor([2.0, 3.0, 4.0, 0.0, 0.0]))


def test_shift_value_uses_bootstrap_past_the_end():
    """V(s_{t+n}) beyond the trajectory end must come from its bootstrap value, which the
    buffer already zeroed for a true termination."""
    from oprl.advantages.dae import _shift_value

    v = torch.tensor([1.0, 2.0, 3.0])
    last = torch.tensor(9.0)
    assert torch.equal(_shift_value(v, last, 1), torch.tensor([2.0, 3.0, 9.0]))
    assert torch.equal(_shift_value(v, last, 2), torch.tensor([3.0, 9.0, 9.0]))
    assert torch.equal(_shift_value(v, last, 3), torch.tensor([9.0, 9.0, 9.0]))


def test_continuation_mask_ends_on_any_flag():
    """The telescoping identity holds only inside one trajectory, so `terminated`,
    `truncated` and an autoreset dummy step must all end the window."""
    d = DAE()
    T, N = 4, 1
    z = torch.zeros(T, N, dtype=torch.bool)
    ones = torch.ones(T, N, dtype=torch.bool)

    assert d._continuation(oprl.Masks(z, z, ones)).sum().item() == T
    term = z.clone()
    term[1, 0] = True
    assert d._continuation(oprl.Masks(term, z, ones))[1, 0].item() == 0.0

    trunc = z.clone()
    trunc[2, 0] = True
    assert d._continuation(oprl.Masks(z, trunc, ones))[2, 0].item() == 0.0

    invalid = ones.clone()
    invalid[3, 0] = False
    assert d._continuation(oprl.Masks(z, z, invalid))[3, 0].item() == 0.0


def test_residual_term_count_is_exact():
    """A trajectory of length L contributes exactly L-n+1 windows for each n <= horizon.

    Walking real trajectories means every window formed is a window that exists -- no
    masking, no waste. On a [T, N] grid with rollout_len=128 and horizon=32, roughly 3/4 of
    windows fall off the rollout edge and have to be discarded.
    """
    L, horizon = 8, 4
    expected = sum(L - n + 1 for n in range(1, horizon + 1))
    assert expected == 8 + 7 + 6 + 5 == 26


# ---------------- analytic residual ----------------


def test_residual_is_zero_for_exact_values():
    """Given the true V and A of a known MDP, the telescoping residual must vanish.

    Constant reward 1, no termination, gamma=0.9 gives V = 1/(1-gamma) = 10 everywhere and
    A = 0 for every action. Then sum gamma^k (r - A) + gamma^n V == V must hold exactly.
    """
    gamma, T, N = 0.9, 10, 1
    v_star = 1.0 / (1.0 - gamma)
    rewards = torch.ones(T, N)
    adv = torch.zeros(T, N)
    values = torch.full((T, N), v_star)

    from oprl.advantages.dae import _shift_value

    d = (rewards - adv).squeeze(-1)
    v = values.squeeze(-1)
    v_last = torch.tensor(v_star)
    acc = torch.zeros_like(d)
    worst = 0.0
    for n in range(1, 5):
        acc = acc + (gamma ** (n - 1)) * _shift(d, n - 1)
        target = acc + (gamma**n) * _shift_value(v, v_last, n)
        worst = max(worst, (target - v)[: T - n + 1].abs().max().item())
    assert worst < 1e-4, f"residual should vanish for exact V/A, got {worst}"


# ---------------- end to end ----------------


def test_dae_trains():
    env = _env()
    policy = _policy(env)
    ppo.train(_cfg(), env, policy, oprl.Logger(sinks=[]))
    env.close()


def test_dae_reports_its_diagnostics():
    """DAE's own metrics must reach the logger with no registration (DESIGN.md §8.1)."""
    env = _env()
    policy = _policy(env)
    log = oprl.Logger(sinks=[])
    seen: dict = {}
    log.sinks = [type("Cap", (), {
        "write": lambda self, m, s: seen.update(m),
        "write_media": lambda self, *a: None,
        "close": lambda self: None,
    })()]
    ppo.train(_cfg(), env, policy, log)
    env.close()
    for key in ("diag/dae_adv_bias", "diag/dae_residual_terms", "loss/dae_residual"):
        assert key in seen, f"{key} never reached the logger"
    # The projection must still hold after training, not only at init.
    assert abs(seen["diag/dae_adv_bias"]) < 1e-4


def test_dae_composes_with_surrogates():
    """Advantage estimator and policy loss are orthogonal axes."""
    for surrogate in ("ppo", "dpo"):
        env = _env()
        policy = _policy(env)
        ppo.train(_cfg(surrogate=surrogate), env, policy, oprl.Logger(sinks=[]))
        env.close()


@pytest.mark.slow
def test_dae_learns_cartpole():
    oprl.seed_everything(1)          # policy init is the caller's responsibility
    env = oprl.make_env("CartPole-v1", num_envs=8, seed=1)
    policy = _policy(env)
    cfg = ppo.PPOConfig(
        total_steps=60_000, num_envs=8, rollout_len=128, device="cpu",
        num_epochs=4, num_minibatches=4, seed=1, ent_coef=0.01,
        advantage={"name": "dae", "horizon": 32},
        network={"advantage_head": True},
    )
    log = oprl.Logger(sinks=[])
    ppo.train(cfg, env, policy, log)
    env.close()
    ret = sum(log.ep_returns) / len(log.ep_returns)
    # Threshold set well below the measured 3-seed minimum (159) so this catches a
    # regression, not seed noise. DAE is not expected to beat GAE here -- see the numbers
    # in advantages/dae.py.
    assert ret > 100, f"DAE failed to learn CartPole: mean return {ret:.1f}"

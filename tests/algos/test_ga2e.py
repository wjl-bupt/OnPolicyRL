"""Tests for the GA2E DIY estimator.

Two things are under test, and they are different claims:

1. **The framework claim.** GA2E is the most demanding thing the `AdvantageEstimator`
   protocol was shaped for (DESIGN.md §4.7): it needs the policy, backpropagation during
   estimation, cross-iteration state, and an epoch-boundary hook. If it plugs in with no
   framework edit, the protocol was designed correctly.
2. **The port claim.** The port must be faithful, which is *not* the same as matching the
   original's numbers -- the original collapsed `terminated` and `truncated` into one mask
   (DESIGN.md §4.1). See `test_two_masks_are_not_collapsed`.
"""

import pytest
import torch

import oprl
from oprl.advantages import get_estimator
from oprl.algos import ppo

GA2E = "./diy/advantages/ga2e.py:GA2E"


def _rollout(num_envs=4, T=48, seed=1):
    env = oprl.make_env("CartPole-v1", num_envs=num_envs, seed=seed)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    buf = oprl.RolloutBuffer(T, num_envs, env.obs_space, env.action_space, "cpu")
    obs = env.reset(seed=seed)
    oprl.collect(env, policy, buf, obs, oprl.Logger(sinks=[]))
    return env, policy, buf


def test_loads_from_path_with_kwargs():
    est = get_estimator({"from": GA2E, "lambda_val": 0.9, "val_every": 3})
    assert est.lambda_val == 0.9 and est.val_every == 3


def test_rejects_bad_refresh_mode():
    with pytest.raises(ValueError, match="refresh"):
        get_estimator({"from": GA2E, "refresh": "sometimes"})


def test_declares_no_extra_buffer_fields():
    """GA2E only re-reads reward/value, so it needs zero schema additions (DESIGN.md §4.3)."""
    est = get_estimator({"from": GA2E})
    assert est.extra_fields == {}
    assert est.resolve_fields(None, None) == {}


def test_gae_matches_the_framework_implementation():
    """The port's internal GAE must agree with `oprl.gae` exactly.

    This is the anchor for every other claim: if the advantage differs from the framework's
    own tested implementation, a performance difference cannot be attributed to lambda
    selection.
    """
    env, policy, buf = _rollout()
    est = get_estimator({"from": GA2E})
    delta, cont, _ = est._delta_cont(buf)
    for lam in (0.0, 0.5, 0.95, 1.0):
        mine = est._gae(delta, cont, lam)
        ref, _ = oprl.gae(buf["reward"][: buf.T], buf.values, buf.bootstrap_value,
                          buf.masks, est.gamma, lam)
        assert torch.allclose(mine, ref, atol=1e-5), f"lambda={lam} diverges from oprl.gae"
    env.close()


def test_two_masks_are_not_collapsed():
    """**The correctness fix.** `terminated` cuts the value bootstrap; `truncated` keeps it.

    The original implementation derived a single `cont` from `episode_starts`, which is the
    field's most widespread on-policy bug. A port that reproduced the original's numbers
    would have had to reproduce this too.
    """
    env, policy, buf = _rollout()
    est = get_estimator({"from": GA2E})

    zero = torch.zeros(buf.T, buf.N, dtype=torch.bool)

    class Masked:
        """Same rollout, one step marked terminated vs truncated."""

        def __init__(self, term, trunc):
            self.terminated, self.truncated = term, trunc
            self.valid = torch.ones(buf.T, buf.N, dtype=torch.bool)

    t = 10
    term, trunc = zero.clone(), zero.clone()
    term[t, :] = True
    buf._buf["terminated"][: buf.T] = term
    buf._buf["truncated"][: buf.T] = zero
    d_term, c_term, _ = est._delta_cont(buf)

    buf._buf["terminated"][: buf.T] = zero
    buf._buf["truncated"][: buf.T] = trunc if trunc.any() else zero
    buf._buf["truncated"][t, :] = True
    d_trunc, c_trunc, _ = est._delta_cont(buf)

    # The bootstrap differs: terminated zeroes gamma*V(s'), truncated keeps it.
    assert not torch.allclose(d_term[t], d_trunc[t]), (
        "terminated and truncated gave the same TD residual -- masks collapsed"
    )
    # The recursion is cut identically by either.
    assert torch.allclose(c_term[t], c_trunc[t])
    env.close()


def test_compute_returns_lambda_inside_the_grid():
    env, policy, buf = _rollout()
    est = get_estimator({"from": GA2E, "val_every": 2})
    adv, ret, diag = est.compute(buf, policy=policy, surrogate=None,
                                 cfg=ppo.PPOConfig())
    assert adv.shape == (buf.T, buf.N) and ret.shape == (buf.T, buf.N)
    assert est.lambda_min <= diag["ga2e/lambda_used"] <= 1.0
    assert diag["ga2e/n_evals"] > 0
    env.close()


def test_compute_leaves_the_policy_untouched():
    """Selection runs ~19 backward passes. None may perturb a parameter or leave a stale
    gradient for the real update to pick up."""
    env, policy, buf = _rollout()
    est = get_estimator({"from": GA2E, "val_every": 2})
    before = [p.detach().clone() for p in policy.parameters()]
    est.compute(buf, policy=policy, surrogate=None, cfg=ppo.PPOConfig())
    assert all(torch.equal(a, b)
               for a, b in zip(before, policy.parameters(), strict=True))
    assert all(p.grad is None for p in policy.parameters()), "left a stale gradient"
    env.close()


def test_needs_the_policy():
    """A pure-function estimator may ignore `policy`; this one must refuse without it."""
    env, policy, buf = _rollout()
    est = get_estimator({"from": GA2E})
    with pytest.raises(ValueError, match="policy"):
        est.compute(buf, policy=None, cfg=ppo.PPOConfig())
    env.close()


def test_trajectory_split_is_disjoint_and_deterministic():
    """Folds must not overlap -- a leaked transition makes the yardstick non-independent --
    and the split must consume no RNG, so it cannot perturb the run's seeding."""
    env, policy, buf = _rollout(num_envs=4, T=64)
    est = get_estimator({"from": GA2E, "val_every": 3})
    _, _, valid = est._delta_cont(buf)
    tr1, val1 = est._trajectory_split(buf, valid)
    tr2, val2 = est._trajectory_split(buf, valid)

    assert torch.equal(tr1, tr2) and torch.equal(val1, val2), "split is not deterministic"
    if est._diag["ga2e/split_degenerate"] == 0.0:
        overlap = set(tr1.tolist()) & set(val1.tolist())
        assert not overlap, f"{len(overlap)} transitions are in both folds"
    env.close()


def test_state_dict_carries_only_the_ema():
    est = get_estimator({"from": GA2E})
    assert est.state_dict() == {"ema_lam": None}
    est.load_state_dict({"ema_lam": 0.77})
    assert abs(est._ema_update(0.77) - 0.77) < 1e-9


def test_ema_smooths_and_stays_in_range():
    est = get_estimator({"from": GA2E, "ema_beta": 0.9, "lambda_min": 0.1})
    first = est._ema_update(0.5)
    assert first == pytest.approx(0.5)          # first call adopts the value
    second = est._ema_update(1.0)
    assert 0.5 < second < 1.0                   # then it lags
    for _ in range(50):
        v = est._ema_update(0.0)
    assert v >= est.lambda_min                  # clipped to the grid


@pytest.mark.parametrize("refresh", ["rollout", "epoch"])
def test_trains_end_to_end(refresh):
    """**The framework claim**: no edit to src/oprl/ was needed for any of this."""
    env = oprl.make_env("CartPole-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    cfg = ppo.PPOConfig(
        total_steps=4 * 32 * 3, num_envs=4, rollout_len=32, device="cpu",
        num_epochs=3, advantage={"from": GA2E, "refresh": refresh, "val_every": 2},
    )

    seen: dict[str, list[float]] = {}

    class Cap(oprl.Logger):
        def record(self, k, v):
            if k.startswith("ga2e/"):
                seen.setdefault(k, []).append(float(v))
            super().record(k, v)

    ppo.train(cfg, env, policy, Cap(sinks=[]))
    env.close()

    # Algorithm-private metrics reach the logger with zero registration (DESIGN.md §8.1).
    assert "ga2e/lambda_used" in seen and "ga2e/lambda_star" in seen
    assert all(0.0 <= v <= 1.0 for v in seen["ga2e/lambda_used"])
    if refresh == "epoch":
        assert "ga2e/lambda_epoch" in seen, "epoch mode never re-selected lambda"


def test_epoch_mode_does_not_move_the_critic_target():
    """In `epoch` mode the advantages are rewritten per epoch, but `returns` must not be --
    otherwise the critic regresses onto a target that moves under it."""
    env, policy, buf = _rollout(T=64)
    est = get_estimator({"from": GA2E, "refresh": "epoch", "val_every": 2})
    cfg = ppo.PPOConfig()
    _, ret0, _ = est.compute(buf, policy=policy, surrogate=None, cfg=cfg)
    buf.returns = ret0
    est.on_epoch_start(1, buf, policy=policy, surrogate=None, cfg=cfg)
    assert torch.equal(buf.returns, ret0), "on_epoch_start moved the critic target"
    env.close()


def test_rollout_mode_ignores_the_epoch_hook():
    """`rollout` mode must treat `on_epoch_start` as a no-op, or it silently pays epoch
    mode's cost."""
    env, policy, buf = _rollout()
    est = get_estimator({"from": GA2E, "refresh": "rollout", "val_every": 2})
    cfg = ppo.PPOConfig()
    adv, _, _ = est.compute(buf, policy=policy, surrogate=None, cfg=cfg)
    buf.advantages = adv
    est.on_epoch_start(1, buf, policy=policy, surrogate=None, cfg=cfg)
    assert torch.equal(buf.advantages, adv)
    env.close()


def test_imports_nothing_from_oprl():
    """Like diy/surrogates/spo.py, this must stand alone -- that is the DIY claim."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "diy/advantages/ga2e.py").read_text(
        encoding="utf-8"
    )
    for line in src.splitlines():
        s = line.strip()
        if s.startswith(("import ", "from ")) and "oprl" in s:
            pytest.fail(f"ga2e.py should not import oprl: {s}")


@pytest.mark.slow
def test_learns_cartpole():
    """GA2E must learn, but it is **not** asserted to beat GAE -- measured, it does not on
    this task. See doc/fix.md for the numbers and why they are recorded rather than tuned.
    """
    env = oprl.make_env("CartPole-v1", num_envs=8, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    cfg = ppo.PPOConfig(
        total_steps=60_000, num_envs=8, rollout_len=128, device="cpu",
        num_epochs=4, num_minibatches=4, seed=1, ent_coef=0.01,
        advantage={"from": GA2E},
    )
    log = oprl.Logger(sinks=[])
    ppo.train(cfg, env, policy, log)
    env.close()
    ret = sum(log.ep_returns) / len(log.ep_returns)
    assert ret > 60, f"GA2E failed to learn CartPole: mean return {ret:.1f}"

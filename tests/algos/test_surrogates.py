"""Surrogate and estimator protocol tests.

Key assertion: **every surrogate must actually learn**, not merely run to completion.
An objective with a numerical mistake still finishes fine -- it just does not learn.
"""

import pytest
import torch

import oprl
from oprl.advantages import ESTIMATORS, get_estimator
from oprl.algos import ppo
from oprl.objectives import SURROGATES, get_surrogate

ALL_SURROGATES = sorted(SURROGATES)


def test_registry_contents():
    """The expected surrogates and estimators are registered."""
    assert set(ALL_SURROGATES) == {
        "ppo", "tr_ppo", "spo", "dpo", "mdpo", "ppo_rpe", "apo", "ppo_kl"
    }
    assert "gae" in ESTIMATORS


def test_get_accepts_name_or_object():
    """A string is only shorthand; passing the object is always equivalent."""
    s = get_surrogate("dpo")
    assert get_surrogate(s) is s
    with pytest.raises(KeyError):
        get_surrogate("nonexistent")
    with pytest.raises(KeyError):
        get_estimator("nonexistent")


@pytest.mark.parametrize("name", ALL_SURROGATES)
def test_surrogate_gradient_direction(name):
    """Basic correctness: a positive advantage must push the action's probability up.

    That is d(loss)/d(logp) < 0, so a sign error in any surrogate is caught here.
    """
    cfg = ppo.PPOConfig(clip_coef=0.2)
    logp_old = torch.zeros(64)
    logp = torch.zeros(64, requires_grad=True)
    adv = torch.ones(64)  # all-positive advantage
    loss, stats = get_surrogate(name)(logp.exp() / 1.0, logp, logp_old, adv, cfg)
    loss.backward()
    assert logp.grad is not None
    assert logp.grad.sum() < 0, f"{name}: wrong gradient direction for advantage > 0"
    assert isinstance(stats, dict)


@pytest.mark.parametrize("name", ALL_SURROGATES)
def test_surrogate_runs(name):
    """Every surrogate runs the full training loop."""
    env = oprl.make_env("CartPole-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    cfg = ppo.PPOConfig(total_steps=4 * 32 * 2, num_envs=4, rollout_len=32,
                        device="cpu", num_epochs=2, surrogate=name)
    ppo.train(cfg, env, policy, oprl.Logger(sinks=[]))
    env.close()


def test_estimator_swappable():
    """Advantage estimators are pluggable via registry dispatch."""
    env = oprl.make_env("CartPole-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    cfg = ppo.PPOConfig(total_steps=4 * 32 * 2, num_envs=4, rollout_len=32,
                        device="cpu", num_epochs=2, advantage="vtrace_free_mc")
    ppo.train(cfg, env, policy, oprl.Logger(sinks=[]))
    env.close()


def test_surrogate_and_estimator_compose():
    """The two protocols are orthogonal and compose freely -- the ablation space."""
    env = oprl.make_env("CartPole-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    cfg = ppo.PPOConfig(total_steps=4 * 32 * 2, num_envs=4, rollout_len=32,
                        device="cpu", num_epochs=2,
                        surrogate="dpo", advantage="vtrace_free_mc")
    ppo.train(cfg, env, policy, oprl.Logger(sinks=[]))
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize("name", ALL_SURROGATES)
def test_surrogate_learns(name):
    """**Core test**: each surrogate must clearly beat random (~20) on CartPole.

    An objective with a numerical error runs without complaint; only this test catches it.
    """
    env = oprl.make_env("CartPole-v1", num_envs=8, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    cfg = ppo.PPOConfig(total_steps=60_000, num_envs=8, rollout_len=128,
                        device="cpu", num_epochs=4, num_minibatches=4,
                        seed=1, surrogate=name)
    log = oprl.Logger(sinks=[])
    ppo.train(cfg, env, policy, log)
    env.close()
    ret = sum(log.ep_returns) / len(log.ep_returns)
    assert ret > 60, f"surrogate '{name}' failed to learn CartPole: mean return {ret:.1f}"


# --------------------------------------------------------------------------- #
#  APO regression tests -- two real bugs lived here
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec",
    ["apo", {"name": "apo"}, {"name": "apo", "uarr_coef": 0.5},
     {"name": "apo", "uarr_coef": "auto"}],
)
def test_apo_uarr_is_active_in_every_spec_form(spec):
    """**Bug 1**: ppo.py dispatched on `cfg.surrogate == "apo"`, so the dict form
    (`{name: apo}`) silently skipped UARR and APO degraded to plain PPO with no error.

    Dispatch is now a capability check on the surrogate object, so every spec form behaves
    the same. Pendulum, because UARR targets continuous action spaces.
    """
    seen: dict[str, list[float]] = {}

    class Cap(oprl.Logger):
        def record(self, k, v):
            if k.startswith("diag/apo") or k == "diag/uarr":
                seen.setdefault(k, []).append(float(v))
            super().record(k, v)

    env = oprl.make_env("Pendulum-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    cfg = ppo.PPOConfig(total_steps=4 * 32 * 3, num_envs=4, rollout_len=32,
                        device="cpu", num_epochs=3, surrogate=spec)
    ppo.train(cfg, env, policy, Cap(sinks=[]))
    env.close()

    assert max(seen["diag/apo_degraded"]) == 0.0, "UARR was skipped"
    assert any(v > 0 for v in seen["diag/uarr"]), (
        "UARR is identically zero -- the anchor is not pi_old (see bug 2)"
    )


def test_apo_uarr_has_a_gradient():
    """**Bug 2**, the worse one: the original resampled from the *current* policy and scored
    those actions with that same policy, so logratio was identically 0 and UARR contributed
    no gradient. The term was decorative.

    Anchoring on pi_old is what makes it real, and a non-zero gradient is the proof.
    """
    import torch

    env = oprl.make_env("Pendulum-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    buf = oprl.RolloutBuffer(32, 4, env.obs_space, env.action_space, "cpu")
    obs = env.reset(seed=1)
    oprl.collect(env, policy, buf, obs, oprl.Logger(sinks=[]))

    sur = get_surrogate("apo")
    sur.on_rollout_end(policy, buf, None)          # anchor under pi_old
    # Move theta so the current policy differs from the anchor.
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 0.05)

    mb = next(buf.iter_minibatches(1))
    sur.prepare(policy, mb, None)
    assert sur._logratio is not None
    assert sur._logratio.abs().max() > 0, "logratio is 0: the anchor is not pi_old"

    uarr = ((sur._logratio.exp() - 1.0) ** 2).mean()
    grads = torch.autograd.grad(uarr, list(policy.parameters()), allow_unused=True)
    total = sum(g.abs().sum().item() for g in grads if g is not None)
    assert total > 0, "UARR has no gradient w.r.t. the policy -- the term does nothing"
    env.close()


def test_apo_degrades_explicitly_when_resampling_is_off():
    """Turning resampling off must be visible in the logs, not silent."""
    sur = get_surrogate({"name": "apo", "resample": False})
    import torch

    cfg = ppo.PPOConfig()
    sur._anchor = None
    _, stats = sur(torch.ones(8), torch.zeros(8), torch.zeros(8), torch.ones(8), cfg)
    assert stats["diag/apo_degraded"] == 1.0


def test_surrogate_hyperparameters_come_from_the_spec():
    """Component hyperparameters live on the component, not on PPOConfig (diy/README.md).

    Eight such fields used to sit on PPOConfig, so every new surrogate widened a dataclass
    shared by all of them.
    """
    assert get_surrogate({"name": "spo", "beta": 2.5}).beta == 2.5
    assert get_surrogate({"name": "dpo", "alpha": 3.0, "beta": 0.4}).alpha == 3.0
    assert get_surrogate({"name": "tr_ppo", "rollback_alpha": 0.9}).rollback_alpha == 0.9
    assert get_surrogate({"name": "ppo_kl", "kl_coef": 0.05}).kl_coef == 0.05

    for removed in ("spo_beta", "dpo_alpha", "mdpo_tk", "rpe_alpha",
                    "apo_uarr_coef", "apo_resample", "rollback_alpha"):
        assert removed not in ppo.PPOConfig.field_names(), (
            f"{removed} is back on PPOConfig; it belongs to a component"
        )


def test_get_surrogate_returns_independent_instances():
    """A surrogate may hold per-minibatch state (APO's anchor), so two lookups must not
    alias one object -- concurrent runs in one process would interfere."""
    assert get_surrogate("apo") is not get_surrogate("apo")


def test_apo_auto_coef_weights_by_the_anchor_action():
    """`uarr_coef="auto"` weights each sample by 0.5 * pi_old(anchor action | s).

    The weight must come from the **anchor** action -- the one UARR constrains -- not from
    the action that happened to be taken in the rollout.
    """
    import torch

    env = oprl.make_env("Pendulum-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    buf = oprl.RolloutBuffer(32, 4, env.obs_space, env.action_space, "cpu")
    obs = env.reset(seed=1)
    oprl.collect(env, policy, buf, obs, oprl.Logger(sinks=[]))

    sur = get_surrogate({"name": "apo", "uarr_coef": "auto"})
    assert sur.auto_coef
    sur.on_rollout_end(policy, buf, None)
    mb = next(buf.iter_minibatches(1))
    sur.prepare(policy, mb, None)

    # The cached anchor logprob is the anchor's, so it must differ from the rollout action's.
    assert sur._anchor_logp_old is not None
    assert not torch.allclose(sur._anchor_logp_old, mb["logprob"]), (
        "the weight is being taken from the rollout action, not the anchor"
    )

    cfg = ppo.PPOConfig()
    _, stats = sur(torch.ones(len(mb["logprob"])), mb["logprob"], mb["logprob"],
                   torch.ones(len(mb["logprob"])), cfg)
    expected = float(0.5 * sur._anchor_logp_old.exp().mean())
    assert abs(stats["diag/uarr_coef"] - expected) < 1e-5
    assert stats["diag/apo_degraded"] == 0.0
    env.close()


def test_apo_zero_coef_means_off_not_auto():
    """`uarr_coef=0.0` must disable the penalty. It previously fell into the dynamic branch,
    so the value that most obviously reads as "off" silently turned something on."""
    import torch

    sur = get_surrogate({"name": "apo", "uarr_coef": 0.0})
    assert not sur.auto_coef
    sur._logratio = torch.full((8,), 0.5)
    sur._anchor_logp_old = torch.zeros(8)
    _, stats = sur(torch.ones(8), torch.zeros(8), torch.zeros(8), torch.ones(8),
                   ppo.PPOConfig())
    assert stats["diag/uarr"] == 0.0
    assert stats["diag/apo_degraded"] == 1.0


def test_apo_degraded_is_a_flag():
    """`diag/apo_degraded` is a 0/1 flag. It briefly reported a probability magnitude, which
    made "is UARR active?" unanswerable from the logs."""
    import torch

    sur = get_surrogate("apo")
    sur._logratio = torch.full((8,), 0.2)
    sur._anchor_logp_old = torch.full((8,), -1.0)
    _, active = sur(torch.ones(8), torch.zeros(8), torch.full((8,), -1.0),
                    torch.ones(8), ppo.PPOConfig())
    assert active["diag/apo_degraded"] == 0.0
    sur._logratio = None
    _, off = sur(torch.ones(8), torch.zeros(8), torch.full((8,), -1.0),
                 torch.ones(8), ppo.PPOConfig())
    assert off["diag/apo_degraded"] == 1.0


def test_apo_rejects_a_negative_coef():
    with pytest.raises(ValueError, match="uarr_coef"):
        get_surrogate({"name": "apo", "uarr_coef": -1.0})

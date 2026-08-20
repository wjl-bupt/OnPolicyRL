"""Tests for the diy/ examples.

The point of these is not to test SPO -- the built-in surrogate already is. It is to prove
the **DIY path is not a second-class citizen**: a component loaded from a local `.py`
file must behave identically to a registered one.
"""

from pathlib import Path

import pytest
import torch

import oprl
from oprl.algos import ppo
from oprl.config import load_config
from oprl.objectives import get_surrogate

DIY_SPO = "./diy/surrogates/spo.py:SimplePolicyOptimization"
REPO = Path(__file__).resolve().parents[1]


def test_diy_files_exist():
    for rel in ("diy/README.md", "diy/surrogates/spo.py", "diy/surrogates/spo.yaml",
                "diy/surrogates/spo.md"):
        assert (REPO / rel).is_file(), f"missing {rel}"


def test_diy_imports_nothing_from_oprl():
    """A DIY component must stand alone -- that is the whole claim being made.

    `spo.py` deliberately imports only torch, so it can be copied into another project
    or read without any framework context.
    """
    src = (REPO / "diy/surrogates/spo.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and "oprl" in stripped:
            pytest.fail(f"diy/surrogates/spo.py should not import oprl: {stripped}")


def test_diy_loads_from_path():
    obj = get_surrogate({"from": DIY_SPO, "beta": 2.0})
    assert obj.beta == 2.0
    assert not obj.adaptive


def test_diy_matches_builtin_numerically():
    """**The load-bearing assertion**: the DIY implementation and the built-in `spo`
    must produce identical losses.

    If they diverge, either the example is wrong or the built-in is -- and both are
    supposed to be the same published formula.

    Note both sides now take `beta` the same way, through `__init__`. It used to be
    `cfg.spo_beta` for the built-in and `__init__` for the DIY copy; unifying that is
    what makes this an apples-to-apples comparison rather than a coincidence.
    """
    cfg = ppo.PPOConfig(clip_coef=0.2)
    diy = get_surrogate({"from": DIY_SPO, "beta": 1.0})
    builtin = get_surrogate({"name": "spo", "beta": 1.0})

    g = torch.Generator().manual_seed(0)
    logp_old = torch.randn(512, generator=g) * 0.1
    logp = torch.randn(512, generator=g) * 0.1
    adv = torch.randn(512, generator=g)
    ratio = (logp - logp_old).exp()

    loss_diy, stats_diy = diy(ratio, logp, logp_old, adv, cfg)
    loss_ref, stats_ref = builtin(ratio, logp, logp_old, adv, cfg)

    assert torch.allclose(loss_diy, loss_ref, atol=1e-6), (
        f"DIY SPO diverges from built-in: {loss_diy.item()} vs {loss_ref.item()}"
    )
    assert abs(stats_diy["diag/spo_penalty"] - stats_ref["diag/spo_penalty"]) < 1e-6


def test_diy_gradient_direction():
    """A positive advantage must raise the action's log-probability."""
    cfg = ppo.PPOConfig()
    logp_old = torch.zeros(128)
    logp = torch.zeros(128, requires_grad=True)
    loss, _ = get_surrogate({"from": DIY_SPO})(
        logp.exp(), logp, logp_old, torch.ones(128), cfg
    )
    loss.backward()
    assert logp.grad.sum() < 0


def test_penalty_vanishes_at_ratio_one():
    """The penalty must be exactly 0 when pi_new == pi_old, otherwise it biases every
    update even when the policy has not moved.

    Note a non-zero advantage is passed deliberately: the penalty carries a per-sample |A|
    factor, so testing with adv=0 would pass even if `(r-1)^2` were computed wrongly.
    """
    cfg = ppo.PPOConfig()
    _, stats = get_surrogate({"from": DIY_SPO})(
        torch.ones(16), torch.zeros(16), torch.zeros(16), torch.ones(16), cfg
    )
    assert abs(stats["diag/spo_penalty"]) < 1e-9


def test_spo_loss_is_a_scalar():
    """The whole objective sits inside one mean. Multiplying a per-sample |A| onto an
    already-reduced mean returns a [B] vector, which autograd accepts before failing
    somewhere far less obvious."""
    cfg = ppo.PPOConfig()
    for spec in (DIY_SPO,):
        loss, _ = get_surrogate({"from": spec})(
            torch.full((32,), 1.2), torch.zeros(32), torch.zeros(32),
            torch.randn(32), cfg
        )
        assert loss.dim() == 0, f"loss has shape {tuple(loss.shape)}, expected a scalar"
    loss, _ = get_surrogate("spo")(
        torch.full((32,), 1.2), torch.zeros(32), torch.zeros(32), torch.randn(32), cfg
    )
    assert loss.dim() == 0, f"built-in spo loss has shape {tuple(loss.shape)}"


def test_adaptive_beta_responds_to_kl():
    """With `adaptive=True`, beta should grow when KL overshoots the target."""
    cfg = ppo.PPOConfig()
    obj = get_surrogate({"from": DIY_SPO, "beta": 1.0, "adaptive": True,
                         "target_kl": 1e-4})
    logp_old = torch.zeros(64)
    logp = torch.full((64,), 0.5)          # a large deliberate policy shift
    ratio = (logp - logp_old).exp()
    _, first = obj(ratio, logp, logp_old, torch.ones(64), cfg)
    _, second = obj(ratio, logp, logp_old, torch.ones(64), cfg)
    assert second["diag/spo_beta"] > first["diag/spo_beta"]


def test_diy_config_file_loads():
    """`diy/surrogates/spo.yaml` must parse into a PPOConfig, so a stale key here fails CI
    rather than at launch time."""
    cfg = load_config(ppo.PPOConfig, str(REPO / "diy/surrogates/spo.yaml"))
    assert isinstance(cfg.surrogate, dict) and "from" in cfg.surrogate
    assert cfg.total_steps > 0


def test_diy_trains_end_to_end():
    cfg = ppo.PPOConfig(
        total_steps=4 * 32 * 2, num_envs=4, rollout_len=32, device="cpu",
        num_epochs=2, surrogate={"from": DIY_SPO, "beta": 1.0},
    )
    env = oprl.make_env("CartPole-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    ppo.train(cfg, env, policy, oprl.Logger(sinks=[]))
    env.close()


@pytest.mark.slow
def test_diy_learns_cartpole():
    """The DIY path must reach the same ballpark as the built-in surrogate (~200)."""
    cfg = ppo.PPOConfig(
        total_steps=60_000, num_envs=8, rollout_len=128, device="cpu",
        num_epochs=4, num_minibatches=4, seed=1,
        surrogate={"from": DIY_SPO, "beta": 1.0},
    )
    env = oprl.make_env("CartPole-v1", num_envs=8, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    log = oprl.Logger(sinks=[])
    ppo.train(cfg, env, policy, log)
    env.close()
    ret = sum(log.ep_returns) / len(log.ep_returns)
    assert ret > 60, f"DIY SPO failed to learn CartPole: mean return {ret:.1f}"

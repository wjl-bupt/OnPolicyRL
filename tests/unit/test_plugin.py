"""Config-driven assembly tests -- acceptance for "the config file is the only
assembly point"."""

import pytest
import torch

import oprl
from oprl.algos import ppo
from oprl.envs import ATARI, MUJOCO, detect_preset, get_preset
from oprl.registry import build, registered, resolve
from oprl.schema import env_symbols, parse_field, schema_from_config

CUSTOM = "./examples/my_components.py"


def test_all_kinds_registered():
    assert "gae" in registered("advantage")
    assert {"ppo", "dpo", "spo"} <= set(registered("policy_loss"))
    assert {"clipped", "mse", "huber"} <= set(registered("value_loss"))
    assert {"mlp", "cnn"} <= set(registered("encoder"))
    assert {"atari", "mujoco", "classic"} <= set(registered("env_preset"))


def test_spec_forms():
    """All four spec forms resolve."""
    assert build("value_loss", "mse") is not None
    assert build("value_loss", {"name": "huber", "delta": 2.0}).delta == 2.0
    obj = build("advantage", {"from": f"{CUSTOM}:MyAdvantage", "lam_start": 0.8})
    assert obj.lam_start == 0.8
    inst = build("value_loss", "mse")
    assert resolve("value_loss", inst)[0] is inst   # instances pass through


def test_unknown_component_suggests_from():
    with pytest.raises(KeyError, match="from"):
        build("advantage", "no_such_estimator")


# ---------------- buffer fields declared in config ----------------


def test_field_from_config():
    f = parse_field({"shape": [4, 2], "dtype": "int64", "sample_op": "whole",
                     "extra_step": True, "doc": "x"})
    assert f.shape == (4, 2) and f.dtype == torch.int64 and f.extra_step
    assert not f.per_sample


def test_field_symbolic_shape():
    """A config may write `n_actions` instead of hard-coding an env-dependent size."""
    import gymnasium as gym

    e = gym.make("CartPole-v1")
    sym = env_symbols(e.observation_space, e.action_space)
    sch = schema_from_config({"probs": {"shape": ["n_actions"]}}, sym)
    assert sch["probs"].shape == (2,)


def test_field_rejects_typo():
    with pytest.raises(ValueError, match="unknown field attributes"):
        parse_field({"shpae": [4]})
    with pytest.raises(ValueError, match="unknown dtype"):
        parse_field({"dtype": "flaot32"})


def test_buffer_extra_from_config():
    env = oprl.make_env("CartPole-v1", num_envs=2, seed=1)
    cfg = ppo.PPOConfig(buffer={"extra": {"probs": {"shape": ["n_actions"]}}})
    extra = ppo.buffer_extra(cfg, oprl.GAE(), env)
    assert "probs" in extra and extra["probs"].shape == (2,)
    env.close()


# ---------------- env presets ----------------


def test_preset_autodetect():
    assert detect_preset("ALE/Pong-v5") is ATARI
    assert detect_preset("HalfCheetah-v5") is MUJOCO
    assert detect_preset("Unknown-v0").name == "raw"   # falls back, but reports it


def test_preset_override():
    p = get_preset({"preset": "atari", "frame_stack": 2}, "ALE/Pong-v5")
    assert p.frame_stack == 2 and p.grayscale      # one option changed, rest preserved


def test_preset_rejects_typo():
    with pytest.raises(ValueError, match="unknown env preset options"):
        get_preset({"preset": "atari", "frame_stak": 2}, "ALE/Pong-v5")


# ---------------- end to end: every component from config ----------------


def test_train_with_all_custom_components():
    """A custom encoder, advantage estimator, policy loss, value loss and buffer field,
    all assembled from configuration with no framework edits."""
    cfg = ppo.PPOConfig(
        total_steps=4 * 32 * 2, num_envs=4, rollout_len=32, device="cpu",
        num_epochs=2,
        advantage={"from": f"{CUSTOM}:MyAdvantage", "lam_start": 0.8},
        surrogate={"from": f"{CUSTOM}:MyPolicyLoss", "beta": 0.3},
        value_loss={"name": "huber", "delta": 2.0},
        network={"encoder": {"from": f"{CUSTOM}:MyEncoder", "width": 32}},
        buffer={"extra": {"probs": {"shape": ["n_actions"]}}},
    )
    env = oprl.make_env("CartPole-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic.from_config(env.obs_space, env.action_space, cfg.network)
    ppo.train(cfg, env, policy, oprl.Logger(sinks=[]))
    env.close()


def test_network_from_config():
    import gymnasium as gym

    e = gym.make("CartPole-v1")
    small = oprl.ActorCritic.from_config(e.observation_space, e.action_space,
                                        {"hidden": [16]})
    big = oprl.ActorCritic.from_config(e.observation_space, e.action_space,
                                       {"hidden": [256, 256]})
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(small) < n(big)


def test_network_rejects_typo():
    import gymnasium as gym

    e = gym.make("CartPole-v1")
    with pytest.raises(ValueError, match="unknown network options"):
        oprl.ActorCritic.from_config(e.observation_space, e.action_space,
                                     {"hiden": [16]})

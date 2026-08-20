"""Tests for the algo registry -- the sixth component kind.

The property under test: **adding an algorithm requires no edit to the framework.** Before
the registry, it meant editing five hardcoded sites across `cli.py` and `algos/__init__.py`,
which made the cheapest-sounding claim in DESIGN.md §4.7 the most expensive one in practice.
"""

from pathlib import Path

import pytest

import oprl
from oprl.algos import ppo, vmpo
from oprl.algos.base import ALGOS, Algo, get_algo, registered_algos

REPO = Path(__file__).resolve().parents[2]


def test_builtins_registered():
    assert {"ppo", "a2c", "vmpo"} <= set(registered_algos())


def test_algo_carries_its_config_class():
    """An algorithm is a (train, config_cls) pair -- the CLI needs both to build `--help`."""
    assert ALGOS["ppo"].config_cls is ppo.PPOConfig
    assert ALGOS["vmpo"].config_cls is vmpo.VMPOConfig
    assert ALGOS["ppo"].train is ppo.train


def test_a2c_is_an_alias_not_a_file():
    """A2C shares PPO's train(); it differs only in hyperparameters (DESIGN.md §4.6)."""
    a2c = ALGOS["a2c"]
    assert a2c.train is ALGOS["ppo"].train
    cfg = a2c.make_config({})
    assert cfg.num_epochs == 1 and cfg.num_minibatches == 1
    assert cfg.clip_coef == float("inf") and cfg.gae_lambda == 1.0


def test_alias_defaults_are_overridable():
    """Precedence: dataclass defaults < algo defaults < caller. An alias sets a default,
    it does not pin a value."""
    cfg = ALGOS["a2c"].make_config({"num_epochs": 7})
    assert cfg.num_epochs == 7


def test_registry_still_rejects_unknown_hyperparameters():
    """Going through the registry must not weaken config strictness."""
    with pytest.raises(ValueError, match="does not recognize"):
        ALGOS["ppo"].make_config({"no_such_hyperparameter": 1})


def test_get_algo_accepts_name_object_and_path():
    a = get_algo("ppo")
    assert get_algo(a) is a
    with pytest.raises(KeyError, match="no algo"):
        get_algo("nonexistent")


def test_get_algo_loads_a_train_function_from_file(tmp_path):
    """**The load-bearing test**: a brand new algorithm in a file outside src/, reachable
    with no framework edit at all."""
    f = tmp_path / "my_algo.py"
    f.write_text(
        "def train(cfg, env, policy, log=None, estimator=None):\n"
        "    train.called = True\n"
        "    return policy\n",
        encoding="utf-8",
    )
    a = get_algo(f"{f}:train")
    assert a.config_cls is ppo.PPOConfig       # documented fallback
    assert callable(a.train)


def test_custom_algo_trains_through_the_registry(tmp_path):
    """A file-based algorithm receives a real config, env and policy and runs."""
    f = tmp_path / "counting_algo.py"
    f.write_text(
        "STEPS = []\n"
        "def train(cfg, env, policy, log=None, estimator=None):\n"
        "    import oprl\n"
        "    buf = oprl.RolloutBuffer(cfg.rollout_len, env.num_envs, env.obs_space,\n"
        "                             env.action_space, 'cpu')\n"
        "    obs = env.reset(seed=cfg.seed)\n"
        "    log = log or oprl.Logger(sinks=[])\n"
        "    obs, steps = oprl.collect(env, policy, buf, obs, log)\n"
        "    STEPS.append(steps)\n"
        "    return policy\n",
        encoding="utf-8",
    )
    a = get_algo(f"{f}:train")
    cfg = a.make_config({"num_envs": 4, "rollout_len": 16, "device": "cpu"})
    env = oprl.make_env("CartPole-v1", num_envs=4, seed=1)
    policy = oprl.ActorCritic(env.obs_space, env.action_space)
    a.train(cfg, env, policy, oprl.Logger(sinks=[]))
    env.close()


def test_algo_appears_in_the_shared_registry():
    """`algo` is a first-class kind, so `oprl components` lists it like the others."""
    import oprl.algos  # noqa: F401
    from oprl.registry import KINDS, registered

    assert "algo" in KINDS
    assert "ppo" in registered("algo")


def test_cli_does_not_hardcode_algorithm_names():
    """Regression guard for the actual defect: `cli.py` used to name ppo/a2c/vmpo in five
    places, so a new algorithm could not be added without editing it."""
    src = (REPO / "src" / "oprl" / "cli.py").read_text(encoding="utf-8")
    for name in ('"vmpo"', "'vmpo'", '"a2c"', "'a2c'"):
        assert name not in src, f"cli.py hardcodes {name}; dispatch through the registry"


def test_duplicate_registration_raises():
    from oprl.algos.base import Algo, register_algo

    with pytest.raises(KeyError, match="already registered"):
        register_algo(Algo("ppo", lambda *a, **k: None, ppo.PPOConfig))


def test_reregistering_an_identical_record_is_idempotent():
    """Module reimport (pytest collection, `importlib.reload`) must not raise."""
    from oprl.algos.base import register_algo

    same = Algo("ppo", ppo.train, ppo.PPOConfig, dict(ALGOS["ppo"].defaults),
                ALGOS["ppo"].note)
    assert register_algo(same) is same

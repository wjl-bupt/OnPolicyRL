"""Config layer tests: preset loading, precedence, error handling.

The key property is that **unknown hyperparameters must raise**. Silently ignoring them
lets you believe you changed a setting when you did not -- one of the hardest experiment
errors to notice.
"""

import pytest

from oprl.algos.ppo import PPOConfig
from oprl.algos.vmpo import VMPOConfig
from oprl.config import available, load_config, load_dict, presets


def test_all_presets_load():
    """Every preset must parse into its Config, which stops configs from rotting."""
    for entry in available():
        algo, name = entry.split(":", 1)
        cls = VMPOConfig if algo == "vmpo" else PPOConfig
        cfg = cls.from_dict(load_dict(name, algo))
        assert cfg.total_steps > 0, entry


def test_presets_are_sparse():
    """A preset lists **only what deviates from the defaults**, never a full copy.

    This guards against bloat: a preset with more than ten keys is probably duplicating
    defaults, and the right question is whether all of them really need to differ.
    """
    for entry in available():
        algo, name = entry.split(":", 1)
        n = len(load_dict(name, algo))
        assert n <= 10, f"{entry} has {n} keys; presets should list only deviations"


def test_note_is_not_a_hyperparameter():
    """`note` is prose for humans and must not leak into hyperparameters."""
    for entry in available():
        algo, name = entry.split(":", 1)
        assert "note" not in load_dict(name, algo)
    assert "CartPole" in presets("ppo")["classic"]


def test_override_order():
    """Precedence: dataclass defaults < preset < keyword overrides."""
    assert PPOConfig().lr == 3e-4                                  # default
    assert load_config(PPOConfig, "classic").ent_coef == 0.01      # preset
    assert load_config(PPOConfig, "classic", lr=1e-5).lr == 1e-5   # override
    # Keys the preset does not mention keep their defaults.
    assert load_config(PPOConfig, "classic").gamma == PPOConfig().gamma


def test_no_preset_means_defaults():
    assert load_dict(None) == {}
    assert load_config(PPOConfig).lr == PPOConfig().lr


def test_unknown_preset_lists_alternatives():
    with pytest.raises(KeyError, match="available"):
        load_dict("does_not_exist", "ppo")


def test_unknown_key_rejected_with_hint():
    """**Core assertion**: a misspelled hyperparameter raises, with a spelling hint."""
    with pytest.raises(ValueError, match="does not recognize"):
        PPOConfig.from_dict({"learning_rate": 1e-4})
    with pytest.raises(ValueError, match="lr"):
        PPOConfig.from_dict({"lrr": 1e-4})


def test_unknown_key_allowed_when_not_strict():
    assert PPOConfig.from_dict({"bogus": 1}, strict=False).lr == 3e-4


def test_roundtrip_save_load(tmp_path):
    """A saved resolved config can be replayed via --config -- the basis of reproducibility."""
    cfg = load_config(PPOConfig, "classic", lr=1.5e-4, seed=42)
    p = tmp_path / "config.yaml"
    cfg.save(p)
    reloaded = load_config(PPOConfig, str(p))
    assert reloaded.lr == 1.5e-4 and reloaded.seed == 42


def test_describe_marks_non_defaults():
    out = load_config(PPOConfig, "classic").describe()
    assert "ent_coef" in out and "*" in out
    assert "differs from default" in out


def test_runtime_fields_not_exported():
    """Runtime fields starting with `_` (e.g. MDPO's _progress) stay out of saved configs."""
    cfg = PPOConfig()
    cfg._progress = 0.5
    assert "_progress" not in cfg.to_dict()

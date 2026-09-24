"""Checkpoint save/load roundtrip -- common/checkpoint.py (L1).

Resume correctness rests on three things surviving a save/load: the weights, the
optimizer moments, and the global RNG stream. Each gets a test.
"""

import numpy as np
import pytest
import torch as th

pytest.importorskip("gymnasium")
import gymnasium as gym

from architecture.actor_critic import build_policy
from common.checkpoint import (fingerprint, latest_checkpoint, load_checkpoint,
                               peek_checkpoint, save_checkpoint)


def _policy():
    return build_policy(gym.spaces.Box(-1, 1, (4,), dtype=np.float32),
                        gym.spaces.Discrete(2))


def _step_once(pol):
    """One optimizer step so weights move and Adam state is non-trivial."""
    o = th.zeros(8, 4)
    logp, _ent, value = pol.evaluate(o, th.zeros(8, dtype=th.long))
    pol.optimizer.zero_grad()
    (logp.mean() + value.mean()).backward()
    pol.optimizer.step()


def test_roundtrip_restores_params_and_progress(tmp_path):
    pol = _policy()
    _step_once(pol)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, pol, global_step=1280, iteration=10,
                    meta={"identity": "abc123"})
    assert path.exists()

    saved = {k: v.clone() for k, v in pol.state_dict().items()}
    with th.no_grad():                       # corrupt the live weights
        for p in pol.parameters():
            p.add_(1.0)

    info = load_checkpoint(path, pol)
    assert info["global_step"] == 1280 and info["iteration"] == 10
    assert info["meta"]["identity"] == "abc123"
    for k, v in pol.state_dict().items():
        assert th.allclose(v, saved[k]), f"param {k} not restored"


def test_optimizer_state_restored_into_fresh_policy(tmp_path):
    pol = _policy()
    _step_once(pol)
    path = tmp_path / "c.pt"
    save_checkpoint(path, pol, global_step=1, iteration=1)

    fresh = _policy()
    load_checkpoint(path, fresh)
    src = pol.optimizer.state_dict()["state"]
    dst = fresh.optimizer.state_dict()["state"]
    assert src.keys() == dst.keys() and len(dst) > 0


def test_rng_state_is_restored(tmp_path):
    pol = _policy()                          # build BEFORE seeding (build consumes rng)
    path = tmp_path / "c.pt"
    th.manual_seed(123)
    save_checkpoint(path, pol, global_step=0, iteration=0)
    a = th.randn(4)
    load_checkpoint(path, pol)               # rewinds the stream to the saved state
    b = th.randn(4)
    assert th.allclose(a, b)


def test_peek_does_not_need_a_policy(tmp_path):
    pol = _policy()
    path = tmp_path / "c.pt"
    save_checkpoint(path, pol, global_step=512, iteration=4)
    peek = peek_checkpoint(path)
    assert peek["global_step"] == 512 and peek["iteration"] == 4


# --------------------------- fingerprint + model pool ---------------------- #


def test_fingerprint_stable_and_sensitive():
    cfg = {"lr": 3e-4, "epsilon": 0.2}
    hw = {"device": "cpu", "torch": "2.0", "machine": "x86_64"}
    fp = fingerprint(cfg, hw)
    assert fingerprint({"epsilon": 0.2, "lr": 3e-4}, hw) == fp    # key order irrelevant
    assert fingerprint({**cfg, "lr": 1e-3}, hw) != fp             # config change
    assert fingerprint(cfg, {**hw, "torch": "2.1"}) != fp         # hardware change


def test_latest_checkpoint_picks_largest_step(tmp_path):
    pool = tmp_path / "model_pools"
    pool.mkdir()
    for step in (128, 2048, 512):                 # out of order on purpose
        (pool / f"model_step-{step}.pt").write_bytes(b"x")
    assert latest_checkpoint(pool).name == "model_step-2048.pt"


def test_latest_checkpoint_empty_or_absent(tmp_path):
    assert latest_checkpoint(tmp_path / "nope") is None       # absent dir
    (tmp_path / "empty").mkdir()
    assert latest_checkpoint(tmp_path / "empty") is None      # empty pool


def test_fingerprint_survives_save_load_roundtrip(tmp_path):
    pol = _policy()
    path = tmp_path / "model_pools" / "model_step-256.pt"
    save_checkpoint(path, pol, global_step=256, iteration=2,
                    meta={"fingerprint": "deadbeef"})
    assert peek_checkpoint(path)["meta"]["fingerprint"] == "deadbeef"
    assert load_checkpoint(path, pol)["meta"]["fingerprint"] == "deadbeef"

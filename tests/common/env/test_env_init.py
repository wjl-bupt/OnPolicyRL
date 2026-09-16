"""Env-builder initialization tests (src/common/env).

What is under test, per the builder contract:

1. the default wrapper stack actually engages (obs shapes/dtypes change in the
   documented way),
2. same seed -> bitwise-identical initial observations,
3. stepping produces batched, finite outputs,
4. the escape hatches work (`use_default_wrappers=False`, backend routing),
5. the envpool interface quirks are pinned: int32 actions, [batch_size] action
   arrays, tokenized MiniGrid mission.

Optional backends (envpool / minigrid / mujoco / ale) are skipped gracefully when
not installed, so the file runs on any machine.
"""

import numpy as np
import pytest
import gymnasium as gym

pytest.importorskip("gymnasium", reason="env builders need gymnasium")

from common.env.make_env import detect_family, make_env  # noqa: E402
from common.env.atari_env import make_atari_env  # noqa: E402
from common.env.minatar_env import make_minatar_env  # noqa: E402

# --------------------------------------------------------------------------- #
#  MinAtar (gymnasium only)
# --------------------------------------------------------------------------- #


def test_minatar_default_stack_is_channels_first_and_bool():
    env = make_minatar_env("MinAtar/Breakout-v1", num_envs=2, seed=0)
    try:
        # [H,W,C]=10x10x4 bool was transposed to [C,H,W]; dtype stays bool --
        # rescaling bool obs by /255 collapses it, so it must never happen here.
        assert env.single_observation_space.shape == (4, 10, 10)
        assert env.single_observation_space.dtype == np.dtype(bool)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (2, 4, 10, 10) and obs.dtype == np.dtype(bool)
    finally:
        env.close()


def test_minatar_same_seed_is_bitwise_identical():
    a = make_minatar_env("MinAtar/Breakout-v1", num_envs=2, seed=7)
    b = make_minatar_env("MinAtar/Breakout-v1", num_envs=2, seed=7)
    try:
        oa, _ = a.reset(seed=7)
        ob, _ = b.reset(seed=7)
        assert np.array_equal(oa, ob)
    finally:
        a.close()
        b.close()


def test_minatar_escape_hatch_returns_channels_last_bare_env():
    env = make_minatar_env("MinAtar/Breakout-v1", num_envs=1,
                           use_default_wrappers=False)
    try:
        assert env.single_observation_space.shape == (10, 10, 4)  # raw [H,W,C]
    finally:
        env.close()


# --------------------------------------------------------------------------- #
#  Atari, gymnasium backend
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(detect_family("ALE/Pong-v5") != "atari", reason="id pattern")
def test_atari_default_stack():
    pytest.importorskip("ale_py")
    env = make_atari_env("ALE/Pong-v5", num_envs=2, seed=0)
    try:
        # preprocessing (gray 84x84) + frame stack 4 define the final shape
        assert env.single_observation_space.shape == (4, 84, 84)
        assert env.single_observation_space.dtype == np.dtype(np.uint8)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (2, 4, 84, 84)
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample())
            assert np.isfinite(obs).all() and reward.shape == (2,)
    finally:
        env.close()


def test_atari_escape_hatch_is_raw_ale():
    pytest.importorskip("ale_py")
    env = make_atari_env("ALE/Pong-v5", num_envs=1, use_default_wrappers=False)
    try:
        assert env.single_observation_space.shape == (210, 160, 3)
    finally:
        env.close()


def test_atari_same_seed_is_bitwise_identical():
    pytest.importorskip("ale_py")
    a = make_atari_env("ALE/Pong-v5", num_envs=2, seed=3)
    b = make_atari_env("ALE/Pong-v5", num_envs=2, seed=3)
    try:
        oa, _ = a.reset(seed=3)
        ob, _ = b.reset(seed=3)
        assert np.array_equal(oa, ob)
    finally:
        a.close()
        b.close()


# --------------------------------------------------------------------------- #
#  Atari, envpool backend
# --------------------------------------------------------------------------- #


def test_atari_envpool_matches_gymnasium_shapes():
    envpool = pytest.importorskip("envpool")
    from common.env.atari_env import make_atari_envpool

    env = make_atari_envpool("Pong-v5", num_envs=2, seed=0)
    try:
        assert env.observation_space.shape == (4, 84, 84)
        obs, _ = env.reset()
        assert obs.shape == (2, 4, 84, 84) and obs.dtype == np.dtype(np.uint8)
        act = np.random.randint(env.action_space.n, size=2).astype("int32")
        out = env.step(act)
        assert len(out) == 5 and out[0].shape == (2, 4, 84, 84)
    finally:
        env.close()


def test_atari_envpool_action_array_contract():
    """envpool accepts ONE [batch_size] int32 array per step; a well-shaped int64
    array is silently converted (do not rely on it -- pass int32).

    WRONG SHAPES are deliberately NOT tested here: envpool does not raise a
    Python exception for them -- it **aborts the whole process** from C++ (absl
    CHECK failure, SIGABRT), so a pytest.raises cannot contain it. Consequence
    for the adapter layer: validate action shape in Python before stepping.
    """
    pytest.importorskip("envpool")
    from common.env.atari_env import make_atari_envpool

    env = make_atari_envpool("Pong-v5", num_envs=2, seed=0)
    try:
        env.reset()
        n = env.action_space.n
        env.step(np.random.randint(n, size=2).astype("int32"))   # the contract
        env.step(np.random.randint(n, size=2))                    # int64 -> converted
    finally:
        env.close()


# --------------------------------------------------------------------------- #
#  MiniGrid, both backends
# --------------------------------------------------------------------------- #


def test_minigrid_gymnasium_dict_obs_has_string_mission():
    pytest.importorskip("minigrid")
    from common.env.minigrid_env import make_minigrid_env

    env = make_minigrid_env("MiniGrid-DoorKey-5x5-v0", num_envs=2, seed=0,
                            obs_mode="dict")
    try:
        space = env.single_observation_space
        assert set(space.spaces) == {"image", "direction", "mission"}
        obs, _ = env.reset(seed=0)
        # partial view is agent_view_size=7 regardless of the 5x5 grid
        assert obs["image"].shape == (2, 7, 7, 3)
        assert isinstance(obs["mission"], (list, tuple, np.ndarray))
    finally:
        env.close()


def test_minigrid_gymnasium_flat_obs_appends_mission():
    pytest.importorskip("minigrid")
    from common.env.minigrid_env import make_minigrid_env

    env = make_minigrid_env("MiniGrid-DoorKey-5x5-v0", num_envs=1, seed=0,
                            obs_mode="flat")
    try:
        # image 5*5*3 + direction 1 = 76; flat obs must be strictly bigger
        # because the mission tokens are appended.
        assert env.single_observation_space.shape[0] > 76
        obs, _ = env.reset(seed=0)
        assert obs.shape == (1, env.single_observation_space.shape[0])
    finally:
        env.close()


def test_minigrid_envpool_tokenizes_mission():
    pytest.importorskip("envpool")
    from common.env.minigrid_env import make_minigrid_envpool

    env = make_minigrid_envpool("MiniGrid-Empty-5x5-v0", num_envs=2, seed=0)
    try:
        assert set(env.observation_space.spaces) == {"image", "direction", "mission"}
        obs, _ = env.reset()
        assert obs["image"].shape == (2, 7, 7, 3)
        assert obs["mission"].shape == (2, 96) and obs["mission"].dtype == np.uint8
        good = np.random.randint(env.action_space.n, size=2).astype("int32")
        out = env.step(good)
        assert len(out) == 5
    finally:
        env.close()


# --------------------------------------------------------------------------- #
#  MuJoCo (gymnasium only; normalization is algorithm-side, not a wrapper)
# --------------------------------------------------------------------------- #


def test_mujoco_is_raw_and_unnormalized():
    pytest.importorskip("mujoco")
    from common.env.mujoco_env import make_mujoco_env

    env = make_mujoco_env("HalfCheetah-v5", num_envs=2, seed=0)
    try:
        s = env.single_observation_space
        # no obs-normalizer wrapper may hide here: bounds stay infinite
        assert np.isinf(s.low).all() and np.isinf(s.high).all()
        obs, _ = env.reset(seed=0)
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample())
        assert np.isfinite(obs).all() and reward.shape == (2,)
    finally:
        env.close()


def test_mujoco_same_seed_is_bitwise_identical():
    pytest.importorskip("mujoco")
    from common.env.mujoco_env import make_mujoco_env

    a = make_mujoco_env("HalfCheetah-v5", num_envs=2, seed=5)
    b = make_mujoco_env("HalfCheetah-v5", num_envs=2, seed=5)
    try:
        oa, _ = a.reset(seed=5)
        ob, _ = b.reset(seed=5)
        assert np.array_equal(oa, ob)
    finally:
        a.close()
        b.close()


# --------------------------------------------------------------------------- #
#  Dispatcher (make_env / detect_family)
# --------------------------------------------------------------------------- #


def test_detect_family():
    assert detect_family("ALE/Pong-v5") == "atari"
    assert detect_family("PongNoFrameskip-v4") == "atari"
    assert detect_family("MinAtar/Breakout-v1") == "minatar"
    assert detect_family("MiniGrid-DoorKey-5x5-v0") == "minigrid"
    assert detect_family("BabyAI-GoToObj-v0") == "minigrid"
    assert detect_family("HalfCheetah-v5") == "mujoco"
    assert detect_family("TotallyUnknown-v9") == "unknown"


def test_make_env_gymnasium_routes_every_family():
    for env_id, shape in [
        ("ALE/Pong-v5", (4, 84, 84)),
        ("MinAtar/Breakout-v1", (4, 10, 10)),
        ("MiniGrid-DoorKey-5x5-v0", None),
        ("HalfCheetah-v5", (17,)),
    ]:
        env = make_env(env_id, backend="gymnasium", num_envs=2, seed=0)
        try:
            assert env.num_envs == 2
            if shape is not None:
                assert env.single_observation_space.shape == shape
        finally:
            env.close()


def test_make_env_envpool_routes_atari_and_minigrid():
    pytest.importorskip("envpool")
    env = make_env("Pong-v5", backend="envpool", num_envs=2, seed=0)
    env.close()          # bare envpool env: no .close guaranteed, but harmless
    env = make_env("MiniGrid-Empty-5x5-v0", backend="envpool", num_envs=2, seed=0)
    env.reset()
    env.close()


def test_make_env_rejects_invalid_backend_family_pairs():
    # mujoco/minatar are gymnasium-only by design (DESIGN_v2 benchmark matrix)
    with pytest.raises(ValueError, match="envpool backend covers"):
        make_env("HalfCheetah-v5", backend="envpool")
    with pytest.raises(ValueError, match="envpool backend covers"):
        make_env("MinAtar/Breakout-v1", backend="envpool")
    with pytest.raises(ValueError, match="not an envpool task"):
        make_env("DefinitelyNotATask-v0", backend="envpool")
    with pytest.raises(ValueError, match="cannot detect the env family"):
        make_env("DefinitelyNotATask-v0", backend="gymnasium")
    with pytest.raises(ValueError, match="backend must be"):
        make_env("ALE/Pong-v5", backend="gym")

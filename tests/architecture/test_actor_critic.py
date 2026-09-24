"""ActorCritic / build_policy tests -- the S0 network axis (architecture/).

The policy is the only S0 slot: `build_policy(obs_space, act_space, cfg)`
assembles it from duck-typed spaces and exposes the doc's Policy contract
(act / evaluate / predict_value / dist). Discrete-vs-continuous lives in the
head, so swapping the action space never touches PPO.
"""

import numpy as np
import pytest
import torch as th

pytest.importorskip("gymnasium")
import gymnasium as gym
from torch.distributions import Categorical, Normal

from architecture.actor_critic import PolicyConfig, build_policy


def test_discrete_policy_contract():
    pol = build_policy(gym.spaces.Box(-1, 1, (4,), dtype=np.float32),
                       gym.spaces.Discrete(3))
    o = th.zeros(5, 4)
    action, logp, value = pol.act(o)
    assert action.shape == (5,) and action.dtype == th.long
    assert logp.shape == (5,) and value.shape == (5,)

    logp2, ent, value2 = pol.evaluate(o, action)
    assert logp2.shape == (5,) and ent.shape == (5,) and value2.shape == (5,)
    # evaluate is the differentiable update path
    (logp2.mean() + value2.mean()).backward()
    assert any(p.grad is not None for p in pol.parameters())

    assert isinstance(pol.dist(o), Categorical)
    assert pol.predict_value(o).shape == (5,)


def test_continuous_policy_contract():
    pol = build_policy(gym.spaces.Box(-1, 1, (6,), dtype=np.float32),
                       gym.spaces.Box(-1, 1, (2,), dtype=np.float32))
    o = th.zeros(5, 6)
    action, logp, value = pol.act(o)
    assert action.shape == (5, 2) and action.dtype == th.float32
    assert logp.shape == (5,)                     # summed over the action dims
    logp2, ent, _ = pol.evaluate(o, action)
    assert logp2.shape == (5,) and ent.shape == (5,)
    assert isinstance(pol.dist(o), Normal)
    # the state-independent log_std is a trainable parameter
    assert any(p is pol.policy_head.log_std for p in pol.parameters())


def test_image_policy_forces_shared_cnn():
    """uint8 image obs -> a shared conv trunk; Nature geometry on 84x84."""
    pol = build_policy(gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8),
                       gym.spaces.Discrete(6), PolicyConfig(features_dim=128))
    assert pol.share
    o = th.zeros(2, 4, 84, 84, dtype=th.uint8)     # crosses as uint8, scaled inside
    action, _, value = pol.act(o)
    assert action.shape == (2,) and value.shape == (2,)


def test_minatar_geometry_is_selected_for_small_grids():
    """The Nature stack underflows below 36px; a 10x10 grid must pick MINATAR."""
    pol = build_policy(gym.spaces.Box(0, 1, (6, 10, 10), dtype=np.float32),
                       gym.spaces.Discrete(6), PolicyConfig(features_dim=64))
    o = th.zeros(3, 6, 10, 10)
    action, _, value = pol.act(o)
    assert action.shape == (3,) and value.shape == (3,)


def test_separate_has_more_params_than_shared():
    obs, act = gym.spaces.Box(-1, 1, (4,), dtype=np.float32), gym.spaces.Discrete(2)
    sep = build_policy(obs, act, PolicyConfig(share=False))
    shared = build_policy(obs, act, PolicyConfig(share=True))
    assert not sep.share and shared.share
    assert sum(p.numel() for p in sep.parameters()) \
        > sum(p.numel() for p in shared.parameters())      # two trunks vs one


def test_policy_head_starts_near_uniform():
    """gain 0.01 on the policy head keeps the initial policy near-uniform (near
    max entropy), so early updates are not dominated by an arbitrary initial
    preference -- the classic cause of a PPO that plateaus. (Default init gives
    O(1) logits and a visibly sub-maximal entropy.)"""
    import math

    pol = build_policy(gym.spaces.Box(-1, 1, (4,), dtype=np.float32),
                       gym.spaces.Discrete(4))
    entropy = pol.dist(th.randn(256, 4)).entropy()
    assert entropy.mean() > math.log(4) - 0.05     # ln 4 ~ 1.386 = uniform over 4


def test_optimizer_covers_all_parameters():
    pol = build_policy(gym.spaces.Box(-1, 1, (4,), dtype=np.float32),
                       gym.spaces.Discrete(2))
    in_opt = {id(p) for group in pol.optimizer.param_groups for p in group["params"]}
    assert in_opt == {id(p) for p in pol.parameters()}


def test_dict_obs_not_wired_yet():
    obs = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (3, 7, 7), dtype=np.uint8)})
    with pytest.raises(NotImplementedError, match="Dict obs"):
        build_policy(obs, gym.spaces.Discrete(3))

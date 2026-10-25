"""Importing an algorithm must not replace another algorithm's policy class."""
import numpy as np
from gymnasium import spaces
from architecture import actor_critic
from onpolicy.ga2e import GA2EConfig, build_policy


def test_global_policy_registration_does_not_change_ga2e(monkeypatch):
    class OtherPolicy(actor_critic.ActorCritic):
        def __init__(self, *args):
            raise AssertionError('Other algorithm policy leaked into GA2E')

    monkeypatch.setattr(actor_critic, '_POLICY_CLS', OtherPolicy)
    policy = build_policy(spaces.Box(-1, 1, (4,), dtype=np.float32),
                          spaces.Discrete(2), actor_critic.PolicyConfig(net_arch=(8,)),
                          GA2EConfig())
    assert type(policy) is actor_critic.ActorCritic
    assert policy.lambda_ema.dtype.is_floating_point
    assert policy.lambda_ema.element_size() == 8


def test_custom_policy_is_selected_explicitly():
    class CustomPolicy(actor_critic.ActorCritic):
        def __init__(self, extractors, head, value, share, cfg):
            super().__init__(extractors, head, value, share)
            self.custom_config = cfg

    cfg = actor_critic.PolicyConfig(net_arch=(8,))
    policy = actor_critic.build_policy(spaces.Box(-1, 1, (4,), dtype=np.float32),
                                      spaces.Discrete(2), cfg, policy_cls=CustomPolicy)
    assert type(policy) is CustomPolicy
    assert policy.custom_config is cfg

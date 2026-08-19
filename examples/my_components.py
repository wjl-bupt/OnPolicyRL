"""Example custom components -- reference them from a config as
{from: ./examples/my_components.py:XXX}.

**Nothing is inherited from the framework and nothing needs registering.** Satisfying
the relevant protocol is enough (structural subtyping).
"""

import torch
import torch.nn as nn


class MyEncoder(nn.Module):
    """A custom encoder. The contract is just an `out_dim` attribute and `forward`."""

    def __init__(self, in_dim: int = 4, width: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.out_dim = width

    def forward(self, x):
        return self.net(x.float())


class MyAdvantage:
    """A custom advantage estimator: GAE whose lambda grows linearly with progress.

    The protocol requires `compute()`; the rest are no-op defaults here (inheriting
    `oprl.BaseEstimator` would supply them for you).
    """

    extra_fields: dict = {}
    extra_policy_outputs: tuple = ()

    def __init__(self, cfg=None, lam_start: float = 0.9, lam_end: float = 0.99):
        self.gamma = getattr(cfg, "gamma", 0.99)
        self.lam_start, self.lam_end = lam_start, lam_end

    def compute(self, buf, policy=None, surrogate=None, cfg=None):
        from oprl import gae

        p = getattr(cfg, "_progress", 0.0)
        lam = self.lam_start + (self.lam_end - self.lam_start) * p
        adv, ret = gae(buf["reward"][: buf.T], buf.values, buf.bootstrap_value,
                       buf.masks, self.gamma, lam)
        return adv, ret, {"diag/my_lambda": lam}

    def critic_loss(self, policy, mb, cfg):
        return None

    def on_epoch_start(self, epoch, buf, policy=None, surrogate=None, cfg=None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass


class MyPolicyLoss:
    """A custom policy loss with signature (ratio, logp, logp_old, adv, cfg) -> (loss, stats)."""

    def __init__(self, beta: float = 0.5):
        self.beta = beta

    def __call__(self, ratio, logp, logp_old, adv, cfg):
        eps = cfg.clip_coef
        clipped = torch.max(-adv * ratio, -adv * ratio.clamp(1 - eps, 1 + eps))
        entropy_like = -self.beta * (ratio * (ratio.clamp_min(1e-8)).log()).mean()
        return clipped.mean() + entropy_like, {"diag/my_reg": float(entropy_like.detach())}

# -*- encoding: utf-8 -*-
"""Actor-Critic architecture -- the S0 network axis (algo_design §9.2 S0).  @Author june

The S0 `network` slot: the Policy the update rule runs on.

Contract consumed by the update rule (doc algo_design §9.2 S8 / §5.1), the ONE
thing PPO needs beyond env + buffer:

    act(obs)            -> (action, logp, value)      rollout   (S1)
    evaluate(obs, act)  -> (logp, entropy, value)     update    (S8, one forward)
    predict_value(obs)  -> value                      bootstrap (S2)
    dist(obs)           -> torch.distributions        optional capability (GA2E/V-MPO)

Discrete-vs-continuous lives HERE, inside the head -- so swapping the action
space is a network-axis change, never a PPO edit. The distribution and its
per-dimension reduction (Categorical: none; DiagGaussian: sum over action dims)
belong to the head, which is why PPO no longer branches on distribution type.

`build_policy(obs_space, act_space, cfg)` is the assembler ("组装器"): duck-typed
spaces (`.shape` / `.n`, same discipline as protocol.buffer.base_schema -- no
gymnasium import), obs dispatched to MLP / CNN, action to the head.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch as th
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical, Distribution, Normal

from architecture.cnn import ImageExtractor
from architecture.init import layer_init
from architecture.layers import MlpExtractor


@dataclass
class PolicyConfig:
    """Network-axis knobs (S0). Structure is code (heads/encoder); these are the
    hyperparameters the assembler reads."""

    net_arch: tuple[int, ...] = (64, 64)     # MLP hidden widths (vector obs)
    activation: type[nn.Module] = nn.Tanh    # MLP activation; CNN always uses ReLU
    share: bool = False                      # separate actor/critic (default); images force shared
    features_dim: int = 512                  # CNN feature width
    ortho_init: bool = True
    log_std_init: float = 0.0                # Box only
    lr: float = 3e-4
    adam_eps: float = 1e-5                   # CleanRL's PPO eps; matters for stability
    optimizer_cls: type = th.optim.Adam


# --------------------------------------------------------------------------- #
#  heads -- own the distribution AND its per-dimension reduction
# --------------------------------------------------------------------------- #


class CategoricalHead(nn.Module):
    """Discrete actions: logits -> Categorical. Action is a scalar, so log_prob
    and entropy are already per-sample (no reduction)."""

    def __init__(self, in_dim: int, n: int, ortho: bool = True):
        super().__init__()
        self.logits = nn.Linear(in_dim, n)
        if ortho:
            layer_init(self.logits, gain=0.01)

    def dist(self, feat: Tensor) -> Categorical:
        return Categorical(logits=self.logits(feat))

    def log_prob(self, dist: Categorical, actions: Tensor) -> Tensor:
        return dist.log_prob(actions)

    def entropy(self, dist: Categorical) -> Tensor:
        return dist.entropy()


class DiagGaussianHead(nn.Module):
    """Continuous actions: mu(feat) + a state-independent log_std parameter ->
    diagonal Normal. log_prob / entropy sum over the action dimensions to a
    per-sample scalar (the reduction PPO used to do inline)."""

    def __init__(self, in_dim: int, act_dim: int, log_std_init: float = 0.0,
                 ortho: bool = True):
        super().__init__()
        self.mu = nn.Linear(in_dim, act_dim)
        if ortho:
            layer_init(self.mu, gain=0.01)
        self.log_std = nn.Parameter(th.ones(act_dim) * log_std_init)

    def dist(self, feat: Tensor) -> Normal:
        return Normal(self.mu(feat), self.log_std.exp())

    def log_prob(self, dist: Normal, actions: Tensor) -> Tensor:
        return dist.log_prob(actions).sum(-1)

    def entropy(self, dist: Normal) -> Tensor:
        return dist.entropy().sum(-1)


# --------------------------------------------------------------------------- #
#  the policy
# --------------------------------------------------------------------------- #


class ActorCritic(nn.Module):
    """obs -> (action distribution, value). Owns its optimizer (set by the
    assembler). Shared trunk for images, separate actor/critic trunks for
    vectors (the CleanRL/SB3 default that reliably solves classic control)."""

    def __init__(self, extractors, policy_head: nn.Module, value_head: nn.Module,
                 share: bool):
        super().__init__()
        self.share = share
        if share:
            self.extractor = extractors                       # a single nn.Module
        else:
            self.pi_extractor, self.vf_extractor = extractors  # a (pi, vf) pair
        self.policy_head = policy_head
        self.value_head = value_head
        self.optimizer: th.optim.Optimizer | None = None      # assembler fills this

    # -- features: one trunk shared, or two independent trunks --------------- #
    def _features(self, obs) -> tuple[Tensor, Tensor]:
        if self.share:
            f = self.extractor(obs)
            return f, f
        return self.pi_extractor(obs), self.vf_extractor(obs)

    def _value(self, feat_v: Tensor) -> Tensor:
        return self.value_head(feat_v).squeeze(-1)

    # -- the contract -------------------------------------------------------- #
    def dist(self, obs) -> Distribution:
        """Optional capability: the current action distribution (estimators such
        as GA2E, or V-MPO's old-dist capture, ask for this)."""
        feat_pi, _ = self._features(obs)
        return self.policy_head.dist(feat_pi)

    def act(self, obs) -> tuple[Tensor, Tensor, Tensor]:
        """Rollout (S1): sample an action, with its log-prob and V(s). Called by
        PPO under no_grad."""
        feat_pi, feat_v = self._features(obs)
        d = self.policy_head.dist(feat_pi)
        action = d.sample()
        return action, self.policy_head.log_prob(d, action), self._value(feat_v)

    def evaluate(self, obs, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Update (S8): the single forward per minibatch -- log-prob and entropy
        of the stored actions under the current policy, plus V(s)."""
        feat_pi, feat_v = self._features(obs)
        d = self.policy_head.dist(feat_pi)
        return (self.policy_head.log_prob(d, actions),
                self.policy_head.entropy(d),
                self._value(feat_v))

    def predict_value(self, obs) -> Tensor:
        """Bootstrap V(s_T) at the rollout boundary (S2)."""
        if self.share:
            return self._value(self.extractor(obs))
        return self._value(self.vf_extractor(obs))


# --------------------------------------------------------------------------- #
#  the assembler
# --------------------------------------------------------------------------- #


def _obs_shape(space) -> tuple[int, ...]:
    return tuple(space.shape) if getattr(space, "shape", None) is not None else ()


def build_policy(obs_space, act_space, cfg: PolicyConfig | None = None) -> ActorCritic:
    """Assemble the S0 network from the spaces. Duck-typed: any object with the
    right `.shape` / `.n` works, so this never imports gymnasium (mirrors
    protocol.buffer.base_schema). Swapping encoder or head happens here -- no PPO
    or algorithm edit."""
    cfg = cfg or PolicyConfig()
    obs_shape = _obs_shape(obs_space)
    if type(obs_space).__name__ == "Dict":
        raise NotImplementedError(
            "Dict obs (e.g. MiniGrid) need a CombinedExtractor -- add one in "
            "architecture/ that runs a sub-encoder per numeric leaf and encodes "
            "the text `mission` leaf; vector and image obs are wired, dict is not."
        )
    is_image = len(obs_shape) == 3
    share = cfg.share or is_image            # images always share the conv trunk

    def make_extractor() -> nn.Module:
        if is_image:
            scale = 255.0 if np.dtype(obs_space.dtype) == np.uint8 else 1.0
            specs = (ImageExtractor.NATURE if min(obs_shape[1], obs_shape[2]) >= 36
                     else ImageExtractor.MINATAR)
            return ImageExtractor(obs_shape, cfg.features_dim, cfg.ortho_init,
                                  scale, specs)
        if len(obs_shape) != 1:
            raise NotImplementedError(
                f"obs shape {obs_shape} is neither vector (1-D) nor image (3-D)")
        return MlpExtractor(obs_shape[0], cfg.net_arch, cfg.activation, cfg.ortho_init)

    if share:
        extractors: object = make_extractor()
        feat_dim = extractors.out_dim
    else:
        pi_ex, vf_ex = make_extractor(), make_extractor()
        extractors = (pi_ex, vf_ex)
        feat_dim = pi_ex.out_dim

    aname = type(act_space).__name__
    if aname == "Discrete":
        head: nn.Module = CategoricalHead(feat_dim, int(act_space.n), cfg.ortho_init)
    elif aname == "Box":
        head = DiagGaussianHead(feat_dim, int(np.prod(act_space.shape)),
                                cfg.log_std_init, cfg.ortho_init)
    else:
        raise NotImplementedError(
            f"action space {aname!r} is not wired (Discrete / Box are); a "
            f"MultiDiscrete / MultiBinary head belongs here."
        )

    value_head = nn.Linear(feat_dim, 1)
    if cfg.ortho_init:
        layer_init(value_head, gain=1.0)

    policy = ActorCritic(extractors, head, value_head, share)
    policy.optimizer = cfg.optimizer_cls(policy.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    return policy

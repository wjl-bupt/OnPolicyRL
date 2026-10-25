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

    architecture: str = "mlp"  # mlp | simba | bronet | spectralnet
    action_head: str = "auto"  # auto categorical/Gaussian, or diffusion (DPPO)
    context_length: int = 16
    attention_heads: int = 4
    transformer_gated: bool = True
    diffusion_steps: int = 10
    diffusion_hidden_dim: int = 256
    diffusion_blocks: int = 2
    diffusion_time_dim: int = 32
    diffusion_min_std: float = 0.1
    diffusion_clip_denoised: float = 1.0
    hidden_dim: int = 256
    num_blocks: int = 2
    actor_hidden_dim: int | None = None
    critic_hidden_dim: int | None = None
    actor_num_blocks: int | None = None
    critic_num_blocks: int | None = None
    expansion: int = 4
    norm_eps: float = 1e-6
    rsnorm: bool | None = None  # auto: enabled for SimBa by family compiler
    net_arch: tuple[int, ...] = (64, 64)     # MLP hidden widths (vector obs)
    activation: type[nn.Module] = nn.Tanh    # MLP activation; CNN always uses ReLU
    share: bool = False                      # separate actor/critic (default); images force shared
    image_encoder: str = "auto"
    image_net_arch: tuple[int, ...] = ()
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
        self.pi_post = nn.Identity()
        self.vf_post = nn.Identity()
        self.policy_head = policy_head
        self.value_head = value_head
        self.optimizer: th.optim.Optimizer | None = None      # assembler fills this

    # -- features: one trunk shared, or two independent trunks --------------- #
    def _features(self, obs) -> tuple[Tensor, Tensor]:
        if self.share:
            f = self.extractor(obs)
            return self.pi_post(f), self.vf_post(f)
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
            return self._value(self.vf_post(self.extractor(obs)))
        return self._value(self.vf_extractor(obs))


# --------------------------------------------------------------------------- #
#  the assembler
# --------------------------------------------------------------------------- #


def _obs_shape(space) -> tuple[int, ...]:
    return tuple(space.shape) if getattr(space, "shape", None) is not None else ()


# architecture/policy.py 里加一个，或者放你新建的 my_policy.py
def policy_impl(cls) -> ActorCritic:
    """Legacy registration; build_policy selects custom policy_cls explicitly."""
    global _POLICY_CLS
    if not callable(cls):
        raise TypeError(f"policy class must be callable, got {cls!r}")
    _POLICY_CLS = cls
    return cls

_POLICY_CLS: type = ActorCritic  

def build_policy(obs_space, act_space, cfg: PolicyConfig | None = None, *, policy_cls=None) -> ActorCritic:
    """Assemble the S0 network from the spaces. Duck-typed: any object with the
    right `.shape` / `.n` works, so this never imports gymnasium (mirrors
    protocol.buffer.base_schema). Swapping encoder or head happens here -- no PPO
    or algorithm edit."""
    cfg = cfg or PolicyConfig()
    from architecture.backbones import ResidualExtractor, validate_backbone
    validate_backbone(cfg)
    obs_shape = _obs_shape(obs_space)
    if type(obs_space).__name__ == "Dict":
        raise NotImplementedError(
            "Dict obs (e.g. MiniGrid) need a CombinedExtractor -- add one in "
            "architecture/ that runs a sub-encoder per numeric leaf and encodes "
            "the text `mission` leaf; vector and image obs are wired, dict is not."
        )
    is_image = len(obs_shape) == 3
    share = cfg.share or (is_image and cfg.architecture != 'causal_transformer')            # images always share the conv trunk

    def residual(in_dim, branch):
        return ResidualExtractor(in_dim, cfg.architecture,
            getattr(cfg, branch + '_hidden_dim') or cfg.hidden_dim,
            getattr(cfg, branch + '_num_blocks') or cfg.num_blocks,
            cfg.expansion, cfg.norm_eps)

    if share and (not is_image or cfg.architecture == 'causal_transformer') and cfg.architecture != 'mlp':
        if ((cfg.actor_hidden_dim or cfg.hidden_dim) != (cfg.critic_hidden_dim or cfg.hidden_dim)
            or (cfg.actor_num_blocks or cfg.num_blocks) != (cfg.critic_num_blocks or cfg.num_blocks)):
            raise ValueError('Shared vector backbone requires identical actor/critic dimensions and blocks')

    def make_extractor(branch='actor') -> nn.Module:
        if cfg.architecture == 'causal_transformer':
            from architecture.causal import CausalExtractor
            return CausalExtractor(obs_shape, obs_space.dtype, cfg, branch)
        if is_image:
            scale = 255.0 if np.dtype(obs_space.dtype) == np.uint8 else 1.0
            specs = (ImageExtractor.NATURE if min(obs_shape[1], obs_shape[2]) >= 36
                     else ImageExtractor.MINATAR)
            if cfg.image_encoder == "minatar_reference":
                specs = ((32, 3, 1), (32, 3, 1))
            elif cfg.image_encoder != "auto":
                raise ValueError(f"Unknown image encoder: {cfg.image_encoder}")
            return ImageExtractor(obs_shape, cfg.features_dim, cfg.ortho_init,
                                  scale, specs)
        if len(obs_shape) != 1:
            raise NotImplementedError(
                f"obs shape {obs_shape} is neither vector (1-D) nor image (3-D)")
        if cfg.architecture != 'mlp':
            return residual(obs_shape[0], branch)
        return MlpExtractor(obs_shape[0], cfg.net_arch, cfg.activation, cfg.ortho_init)

    if share:
        extractors: object = make_extractor()
        feat_dim = extractors.out_dim
    else:
        pi_ex, vf_ex = make_extractor('actor'), make_extractor('critic')
        extractors = (pi_ex, vf_ex)
        feat_dim = pi_ex.out_dim

    pi_dim = feat_dim
    vf_dim = feat_dim if share else vf_ex.out_dim
    post = None
    if is_image and cfg.architecture in ('simba','bronet','spectralnet'):
        post = (residual(feat_dim, 'actor'), residual(feat_dim, 'critic'))
        pi_dim, vf_dim = post[0].out_dim, post[1].out_dim
    if is_image and cfg.image_net_arch and cfg.architecture == 'mlp':
        post = (MlpExtractor(feat_dim, cfg.image_net_arch, cfg.activation, cfg.ortho_init),
                MlpExtractor(feat_dim, cfg.image_net_arch, cfg.activation, cfg.ortho_init))
        pi_dim = vf_dim = post[0].out_dim

    aname = type(act_space).__name__
    if aname == "Discrete":
        head: nn.Module = CategoricalHead(pi_dim, int(act_space.n), cfg.ortho_init)
    elif aname == "Box":
        head = DiagGaussianHead(pi_dim, int(np.prod(act_space.shape)),
                                cfg.log_std_init, cfg.ortho_init)
    else:
        raise NotImplementedError(
            f"action space {aname!r} is not wired (Discrete / Box are); a "
            f"MultiDiscrete / MultiBinary head belongs here."
        )

    value_head = nn.Linear(vf_dim, 1)
    if cfg.ortho_init:
        layer_init(value_head, gain=1.0)

    # Algorithm-specific policy classes must not leak through module imports.
    policy = (ActorCritic(extractors, head, value_head, share) if policy_cls is None
              else policy_cls(extractors, head, value_head, share, cfg))
    if post is not None:
        policy.pi_post, policy.vf_post = post
    if cfg.action_head == 'diffusion':
        if aname != 'Box' or len(act_space.shape) != 1:
            raise ValueError('Diffusion requires a flat continuous Box action space')
        if not (np.allclose(act_space.low,-1.) and np.allclose(act_space.high,1.)):
            raise ValueError('Diffusion expects normalized [-1,1] actions; use action_mode=rescale')
        from architecture.diffusion import DiffusionPolicy
        policy = DiffusionPolicy(policy, pi_dim, int(np.prod(act_space.shape)), cfg)
    policy.optimizer = cfg.optimizer_cls(policy.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    return policy

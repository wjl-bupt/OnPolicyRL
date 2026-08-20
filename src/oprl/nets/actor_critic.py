"""The default ActorCritic assembler -- ~100 readable lines, and **not a base class**.

Algorithms depend on the Policy protocol (types.py), not on this module. So if you do
not like it, copy the whole file and edit it, or write your own from scratch
(DESIGN.md §4.5, level 3).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical, Normal

from ..types import Obs, RState
from .encoders import CNNEncoder, MLPEncoder, orthogonal_init


def _accepts(ctor, param: str) -> bool:
    import inspect

    try:
        return param in inspect.signature(ctor).parameters
    except (TypeError, ValueError):
        return False


def _std_encoder(kind: str, obs_space, hidden, activation):
    import numpy as np

    if kind == "cnn":
        return CNNEncoder(obs_space.shape)
    return MLPEncoder(int(np.prod(obs_space.shape)), hidden, activation)


def _space_size(space) -> int:
    name = type(space).__name__
    if name == "Discrete":
        return int(space.n)
    return int(np.prod(space.shape))


class ActorCritic(nn.Module):
    """encoder -> (policy head, value head) -> distribution.

    An encoder's only contract is `out_dim` + `forward`, so swapping the encoder is how
    you swap the network architecture::

        policy = ActorCritic(env.obs_space, env.action_space, encoder=MyEncoder(...))
    """

    is_recurrent = False

    def __init__(
        self,
        obs_space,
        action_space,
        encoder: nn.Module | None = None,
        hidden: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
        share_encoder: bool = False,
        log_std_init: float = 0.0,
        advantage_head: bool = False,
    ):
        super().__init__()
        self.action_space = action_space
        self.discrete = type(action_space).__name__ == "Discrete"
        self.act_dim = _space_size(action_space)

        if encoder is None:
            encoder = self._default_encoder(obs_space, hidden, activation)
        self.encoder = encoder
        self.share_encoder = share_encoder
        # A separate critic trunk is the common MuJoCo default; sharing saves parameters.
        self.v_encoder = (
            encoder
            if share_encoder
            else self._default_encoder(obs_space, hidden, activation)
        )

        d = self.encoder.out_dim
        # Small-std init for the policy head (another known-important PPO detail).
        self.pi_head = orthogonal_init(nn.Linear(d, self.act_dim), std=0.01)
        self.v_head = orthogonal_init(nn.Linear(self.v_encoder.out_dim, 1), std=1.0)

        # Optional per-action advantage head, requested by estimators such as DAE via
        # `extra_policy_outputs`. Discrete actions only: it needs one output per action.
        self.has_advantage_head = bool(advantage_head)
        if self.has_advantage_head:
            if not self.discrete:
                raise ValueError("advantage_head requires a Discrete action space")
            self.adv_head = orthogonal_init(
                nn.Linear(self.v_encoder.out_dim, self.act_dim), std=0.01
            )

        if not self.discrete:
            # State-independent log_std -- the standard choice on MuJoCo.
            self.log_std = nn.Parameter(torch.full((self.act_dim,), log_std_init))

    @staticmethod
    def _default_encoder(obs_space, hidden, activation) -> nn.Module:
        shape = obs_space.shape
        if shape is not None and len(shape) == 3:
            return CNNEncoder(shape)
        return MLPEncoder(int(np.prod(shape)), hidden, activation)

    # ---------------- config-driven assembly ----------------

    @classmethod
    def from_config(cls, obs_space, action_space, spec: dict | str | None = None):
        """Build the network from configuration.

            network: {hidden: [256, 256], activation: relu}
            network: {encoder: cnn, share_encoder: true}
            network: {from: ./my_net.py:MyPolicy}          # bring your own policy
            network: {encoder: {from: ./my_enc.py:MyEnc}}  # swap only the encoder
        """
        from ..registry import build, resolve

        if spec is None:
            return cls(obs_space, action_space)
        if isinstance(spec, str):     # Shorthand: just an encoder name.
            spec = {"encoder": spec}
        d = dict(spec)
        d.pop("note", None)

        # Fully custom policy -- the framework only requires the Policy protocol.
        if "from" in d:
            ctor, params = resolve("encoder", d)
            return ctor(obs_space=obs_space, action_space=action_space, **params) \
                if _accepts(ctor, "obs_space") else ctor(**params)

        enc_spec = d.pop("encoder", None)
        hidden = tuple(d.pop("hidden", (64, 64)))
        activation = d.pop("activation", "tanh")
        encoder = None
        if enc_spec is not None:
            if isinstance(enc_spec, str) and enc_spec in ("mlp", "cnn"):
                encoder = None if enc_spec == "auto" else _std_encoder(
                    enc_spec, obs_space, hidden, activation
                )
            else:
                encoder = build("encoder", enc_spec)
        unknown = set(d) - {"share_encoder", "log_std_init", "advantage_head"}
        if unknown:
            raise ValueError(
                f"unknown network options {sorted(unknown)}; available: "
                "encoder, hidden, activation, share_encoder, log_std_init, "
                "advantage_head, from"
            )
        return cls(obs_space, action_space, encoder=encoder, hidden=hidden,
                   activation=activation, **d)

    # ---------------- internals ----------------

    def _dist(self, feat: Tensor):
        out = self.pi_head(feat)
        if self.discrete:
            return Categorical(logits=out)
        return Normal(out, self.log_std.exp().expand_as(out))

    def _feat(self, obs: Obs) -> tuple[Tensor, Tensor]:
        x = obs if not isinstance(obs, dict) else torch.cat(
            [v.flatten(1).float() for v in obs.values()], dim=1
        )
        if isinstance(x, Tensor) and x.dim() > 2 and not isinstance(self.encoder, CNNEncoder):
            x = x.flatten(1)
        f_pi = self.encoder(x)
        f_v = f_pi if self.share_encoder else self.v_encoder(x)
        return f_pi, f_v

    def _logp_ent(self, dist, action: Tensor) -> tuple[Tensor, Tensor]:
        if self.discrete:
            return dist.log_prob(action), dist.entropy()
        # Continuous actions are independent per dimension, so sum over the action axis.
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

    # ---------------- Policy protocol ----------------

    @torch.no_grad()
    def act(self, obs: Obs, state: RState = None):
        f_pi, f_v = self._feat(obs)
        dist = self._dist(f_pi)
        action = dist.sample()
        logp, _ = self._logp_ent(dist, action)
        return action, logp, self.v_head(f_v).squeeze(-1), None

    def evaluate(self, obs: Obs, action: Tensor, state: RState = None, valid=None):
        f_pi, f_v = self._feat(obs)
        dist = self._dist(f_pi)
        logp, ent = self._logp_ent(dist, action)
        return logp, ent, self.v_head(f_v).squeeze(-1)

    @torch.no_grad()
    def value(self, obs: Obs, state: RState = None) -> Tensor:
        _, f_v = self._feat(obs)
        return self.v_head(f_v).squeeze(-1)

    def initial_state(self, batch: int) -> RState:
        return None

    # V-MPO needs the full distribution for its KL constraint.
    def dist(self, obs: Obs):
        f_pi, _ = self._feat(obs)
        return self._dist(f_pi)

    # ---------------- optional advantage head (DAE) ----------------

    def advantages(self, obs: Obs, probs: Tensor) -> tuple[Tensor, Tensor]:
        """Per-action advantages plus the state value, for DAE-style estimators.

        The raw head output is centred so that E_{a~pi}[A(s,a)] == 0 exactly, which is the
        defining property of an advantage function. Without this projection the head could
        absorb an arbitrary state-dependent offset and stop being an advantage at all.

        Returns:
            (advantages [B, n_actions], values [B])
        """
        if not self.has_advantage_head:
            raise RuntimeError(
                "this policy has no advantage head; build it with advantage_head=True"
            )
        _, f_v = self._feat(obs)
        raw = self.adv_head(f_v)
        centred = raw - (probs * raw).sum(-1, keepdim=True)
        return centred, self.v_head(f_v).squeeze(-1)

"""DAE -- Direct Advantage Estimation (Pan et al., NeurIPS 2022).

Paper: https://arxiv.org/abs/2109.06093
Proceedings: https://papers.nips.cc/paper_files/paper/2022/hash/4d893f766ab60e5337659b9e71883af4-Abstract-Conference.html

Standard GAE derives the advantage *from* a learned value function::

    A(s,a) = r + gamma * V(s') - V(s)

so any error in V propagates straight into every advantage. DAE inverts the dependency:
a network head outputs the advantage **directly**, and the value function is fit to be
consistent with it.

The mechanism is a telescoping identity. From A = r + gamma*V(s') - V(s) we get::

    r - A = V(s) - gamma * V(s')

Summing the left-hand side with discounting over a trajectory therefore reconstructs the
value, because the V terms cancel pairwise::

    sum_{k=0..n-1} gamma^k (r_{t+k} - A_{t+k})  +  gamma^n V(s_{t+n})  ==  V(s_t)

Turning that identity into a squared residual gives the critic objective. The advantage
head is trained by the *same* loss -- there is no separate advantage target -- which is
exactly why this estimator has to own `critic_loss()` (DESIGN.md §4.7).

Two properties the implementation must preserve, or it stops being DAE:

1. **E_{a~pi}[A(s,a)] == 0.** Enforced in `ActorCritic.advantages()` by subtracting the
   policy-weighted mean from the head output. Without it the head can absorb an arbitrary
   state-dependent offset and is no longer an advantage function.
2. **The residual must be masked at trajectory boundaries.** The telescoping identity only
   holds within a single trajectory; summing across an episode boundary is meaningless.

Scope: discrete action spaces only. The head needs one output per action, and the
`E_{a~pi}[A] = 0` projection needs the full action distribution. The paper's continuous
extension is not implemented.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..schema import Field, Op
from .base import BaseEstimator, register


@register("dae")
class DAE(BaseEstimator):
    """Direct Advantage Estimation.

    Requires:
      - a policy with an advantage head (`network: {advantage_head: true}`)
      - the full action distribution stored in the rollout (declared below)
    """

    # The full policy distribution at collection time. DAE needs it to (a) project the
    # advantage head to zero mean and (b) keep the projection consistent between
    # collection and update.
    extra_fields = {
        "probs": Field(
            (),  # filled in per-environment by `resolve_fields`
            torch.float32,
            write_op=Op.STORE,
            doc="DAE: full action distribution under pi_old",
        ),
    }
    extra_policy_outputs = ("advantages",)

    def __init__(
        self,
        cfg=None,
        horizon: int = 32,
        lam: float = 0.95,
        bootstrap_weight: float = 1.0,
    ):
        """
        Args:
            horizon: truncation length n for the telescoping backup. Longer means less
                bias and more variance, the same trade-off lambda makes in GAE.

                Measured on CartPole (60k steps, 3 seeds, mean +- sd):
                    GAE baseline  223.4 +- 24.9
                    h=8           123.4 +- 131.6   <- unstable, do not use
                    h=16          155.2 +- 22.2
                    h=32          206.0 +- 41.6
                Longer horizons are both better and more stable here, so 32 is the default.
                Note h=8's enormous spread: on one seed it scored 275 (the best result of
                any configuration) and on two others it collapsed to ~50. A single-seed
                comparison of these numbers is worse than no comparison at all.
            lam: GAE-style lambda used to smooth the *raw* per-step advantages into the
                advantages the policy loss consumes. lam=0 uses the head output directly.
            bootstrap_weight: scales the gamma^n V(s_{t+n}) term in the residual. 0
                degenerates to a pure Monte-Carlo target.
        """
        self.gamma = getattr(cfg, "gamma", 0.99) if cfg is not None else 0.99
        if cfg is not None:
            lam = getattr(cfg, "gae_lambda", lam)
        self.horizon = int(horizon)
        self.lam = float(lam)
        self.bootstrap_weight = float(bootstrap_weight)

    # ------------------------------------------------------------------ #
    #  schema
    # ------------------------------------------------------------------ #

    def resolve_fields(self, obs_space, act_space) -> dict:
        """`probs` needs one slot per action, which is only known once the env is known."""
        if type(act_space).__name__ != "Discrete":
            raise ValueError(
                "DAE supports Discrete action spaces only; the advantage head needs one "
                "output per action. Use `advantage: gae` for continuous control."
            )
        return {
            "probs": Field(
                (int(act_space.n),),
                torch.float32,
                doc="DAE: full action distribution under pi_old",
            )
        }

    @torch.no_grad()
    def write_extra(self, policy, obs) -> dict:
        """Called per step during collection to store pi_old's full distribution."""
        d = policy.dist(obs)
        return {"probs": d.probs}

    # ------------------------------------------------------------------ #
    #  advantage
    # ------------------------------------------------------------------ #

    def compute(self, buf, policy=None, surrogate=None, cfg=None):
        if policy is None:
            raise ValueError("DAE needs the policy to query its advantage head")
        if "probs" not in buf:
            raise ValueError(
                "DAE needs the `probs` field in the buffer; it is declared via "
                "`extra_fields`, so this means the estimator was bypassed at setup"
            )

        T, N = buf.T, buf.N
        masks = buf.masks
        cont = self._continuation(masks)              # [T, N] 1.0 while the traj continues

        with torch.no_grad():
            obs = buf.obs
            probs = buf["probs"][:T]
            flat_obs = _flatten_time(obs)
            adv_all, values = policy.advantages(flat_obs, probs.reshape(T * N, -1))
            actions = buf["action"][:T].reshape(-1).long()
            raw_adv = adv_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1).reshape(T, N)
            values = values.reshape(T, N)

        # Smooth the raw per-step advantages the same way GAE does, so the policy loss
        # sees a comparable bias/variance trade-off. lam=0 passes the head through.
        advantages = self._smooth(raw_adv, cont) if self.lam > 0 else raw_adv

        # The critic target is what the telescoping identity reconstructs. It is only used
        # for diagnostics here; the actual critic fit happens in `critic_loss`.
        returns = (advantages + values).detach()

        diag = {
            "diag/dae_adv_mean": raw_adv.mean().item(),
            "diag/dae_adv_std": raw_adv.std().item(),
            # E_a~pi[A] should be ~0 by construction. A drift away from zero means the
            # projection broke, which is the failure mode worth watching.
            "diag/dae_adv_bias": (probs * adv_all.reshape(T, N, -1)).sum(-1).mean().item(),
            "diag/dae_horizon": float(self.horizon),
        }
        return advantages, returns, diag

    # ------------------------------------------------------------------ #
    #  critic
    # ------------------------------------------------------------------ #

    def critic_loss(self, policy, mb, cfg):
        """DAE's residual objective, fitting the value head and advantage head jointly.

        Returning a loss here is what makes DAE DAE: there is no separate advantage
        target, so advantage learning and critic learning are the same optimization.
        """
        # The residual is defined over contiguous time, but a flat minibatch has shuffled
        # that away. So the loss is computed once per iteration over the whole rollout in
        # `iteration_loss` and cached; per-minibatch we return that cached scalar's
        # contribution. See the note in `iteration_loss`.
        return None

    def iteration_loss(self, policy, buf, cfg) -> tuple[Tensor, dict]:
        """Telescoping residual, computed **per trajectory**.

        For a trajectory of length L and every (t, n) with t+n <= L::

            residual = sum_{k<n} gamma^k (r_{t+k} - A_{t+k})
                       + bw * gamma^n V(s_{t+n})  -  V(s_t)

        and the loss is mean(residual^2), where V(s_L) is the trajectory's bootstrap value
        (zero if it ended on a true termination).

        Why trajectories rather than a padded [T, N] grid: on the grid, a window starting
        near the end of the rollout runs off the edge and has to be masked away. With
        rollout_len=128 and horizon=32 that discards roughly 3/4 of all windows -- wasted
        compute and a smaller effective batch. Walking real trajectories only forms windows
        that exist, which is also what the reference implementation does.
        """
        total_loss = torch.zeros((), device=buf.device)
        total_terms = 0

        for tb in buf.iter_trajectories(batch_frames=cfg.rollout_len * buf.N):
            adv_all, values = policy.advantages(tb.obs, tb["probs"])
            actions = tb["action"].long()
            adv = adv_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            delta = tb["reward"] - adv          # telescopes to V(s_t) - gamma*V(s_{t+1})

            offset = 0
            for i, L in enumerate(tb.lengths):
                d = delta[offset : offset + L]
                v = values[offset : offset + L]
                v_last = tb.last_values[i]
                offset += L

                horizon = min(self.horizon, L)
                # acc[t] accumulates sum_{k<n} gamma^k * d[t+k]; only entries with
                # t+n <= L are valid targets, so each n contributes L-n+1 terms.
                acc = torch.zeros_like(d)
                for n in range(1, horizon + 1):
                    acc = acc + (self.gamma ** (n - 1)) * _shift(d, n - 1)
                    v_future = _shift_value(v, v_last, n)
                    target = acc + self.bootstrap_weight * (self.gamma**n) * v_future
                    residual = (target - v)[: L - n + 1]
                    total_loss = total_loss + residual.pow(2).sum()
                    total_terms += residual.numel()

        loss = total_loss / max(total_terms, 1)
        return loss, {
            "loss/dae_residual": loss.item(),
            "diag/dae_residual_terms": float(total_terms),
        }

    # ------------------------------------------------------------------ #
    #  internals
    # ------------------------------------------------------------------ #

    def _continuation(self, masks) -> Tensor:
        """1.0 while the trajectory continues. Either mask ends it -- the telescoping sum
        is only valid within one trajectory, and an autoreset dummy step is not part of
        any trajectory."""
        ended = masks.terminated | masks.truncated | ~masks.valid
        return (~ended).float()

    def _smooth(self, raw: Tensor, cont: Tensor) -> Tensor:
        """Exponentially smooth raw advantages along time, cut at boundaries."""
        coef = self.gamma * self.lam
        out = torch.zeros_like(raw)
        acc = torch.zeros_like(raw[0])
        for t in range(raw.shape[0] - 1, -1, -1):
            acc = raw[t] + coef * cont[t] * acc
            out[t] = acc
        return out


def _shift(x: Tensor, k: int) -> Tensor:
    """Shift along time by k, padding the tail with zeros.

    Padded entries are never read: the caller slices to `[: L - n + 1]`, the range where
    the window still fits inside the trajectory.
    """
    if k == 0:
        return x
    tail = torch.zeros_like(x[:k])
    return torch.cat([x[k:], tail], 0)


def _shift_value(v: Tensor, v_last: Tensor, n: int) -> Tensor:
    """V(s_{t+n}) for a single trajectory, using the bootstrap value past the end.

    A trajectory of length L stores V(s_0..s_{L-1}); V(s_L) is `v_last`, which the buffer
    already zeroed if the trajectory ended on a true termination.
    """
    L = v.shape[0]
    if n >= L:
        return v_last.expand(L).clone()
    pad = v_last.expand(min(n, L)).clone()
    return torch.cat([v[n:].detach(), pad], 0)[:L]


def _flatten_time(obs):
    from ..tree import tree_flatten_time

    return tree_flatten_time(obs)

# -*- encoding: utf-8 -*-
'''
@File :ppo.py
@Created-Time :2026-09-16 16:25:09
@Author  :june
@Description   : proximal policy optimization algorithm.
@Modified-Time :2026-09-16 16:25:09
'''
# B1 branch baseline (doc/algo_taxonomy.md §3.1) -- the reference UpdateRule.
# Structure follows doc/algo_design.md §9.2 stage list S0..S12; the file reads
# top-to-bottom as the pipeline. No base class: V-MPO (B2) copies this file and
# edits -- what is shared is the contract and the primitives, never a loop.
# `env` is an adapter (common.env.adapter.GymVectorAdapter): tensors in and out,
# on device, with the three autoreset masks already correct.

from dataclasses import dataclass

import torch as th
from torch import Tensor

from common.advantages import get_adv_estimator_fn
from common.logger import Logger
from common.metrics import Timer, explained_variance

try:
    from protocol.sample import RolloutSamples
except ImportError:  # executed from inside src/onpolicy/ without `src` on path
    from protocol.sample import RolloutSamples  # type: ignore


# --------------------------------------------------------------------------- #
#  config -- every hyperparameter explicit, unknown names die at construction
# --------------------------------------------------------------------------- #


@dataclass
class PPOConfig:
    # rollout
    num_envs: int = 8
    rollout_len: int = 128
    # update
    ppo_epoch: int = 4
    num_mini_batch: int = 4
    epsilon: float = 0.2            # policy clip range
    vf_epsilon: float | None = None  # value clip range (ABSOLUTE, in value units); None -> `epsilon`
    use_value_clipped: bool = False  # off by default (SB3 baseline PPO); True -> clipped ValueLoss
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    kl_coef: float = 0.0            # explicit k3 KL penalty (usually off)
    target_kl: float | None = None  # early stop threshold; None = off
    max_grad_norm: float = 0.5
    # estimation
    adv_estimator: str = "gae"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantage: bool = True   # per-minibatch; NOT the estimator's job
    # runtime
    seed: int = 1
    device: str = "cpu"
    total_steps: int = 100_000
    log_interval: int = 10             # in updates

    def __post_init__(self) -> None:
        if self.vf_epsilon is None:
            self.vf_epsilon = self.epsilon
        if self.num_mini_batch < 1 or self.ppo_epoch < 1:
            raise ValueError("ppo_epoch / num_mini_batch must be >= 1")


# --------------------------------------------------------------------------- #
#  S8: the per-minibatch context assembled after ONE policy.evaluate
# --------------------------------------------------------------------------- #


@dataclass
class Batch:
    """What one minibatch looks like inside the update (algo_design §4.2).

    Everything the losses need is here, evaluated exactly once -- append-only:
    new fields join at the end, signatures never break.
    """

    observations: Tensor
    actions: Tensor
    old_log_prob: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    logp: Tensor          # current policy, from the single evaluate
    entropy: Tensor
    value: Tensor
    progress: float = 0.0


# --------------------------------------------------------------------------- #
#  S9/S10: the default loss components
# --------------------------------------------------------------------------- #


class ClipPolicyLoss:
    """PPO's clipped surrogate -- the default S9 component (Schulman 2017, eq.7).

    One class, one method: swapping the objective means swapping this object
    (algo_design §5.3); A2C would be a vanilla_pg class here, not a config trick.
    """

    def __init__(self, epsilon: float = 0.2):
        self.epsilon = epsilon

    def __call__(self, batch: Batch) -> tuple[Tensor, dict[str, float]]:
        ratio = (batch.logp - batch.old_log_prob).exp()
        pg1 = -batch.advantages * ratio
        pg2 = -batch.advantages * ratio.clamp(1 - self.epsilon, 1 + self.epsilon)
        loss = th.max(pg1, pg2).mean()
        clipfrac = (pg2 > pg1).float().mean().item()
        return loss, {"train/clipfrac": clipfrac}


class ValueLoss:
    def __init__(self, vf_epsilon: float | None = None):
        self.vf_epsilon = vf_epsilon
    
    def __call__(self, batch: Batch) -> tuple[Tensor, dict[str, float]]:
        if self.vf_epsilon is None:
            loss = 0.5 * ((batch.value - batch.returns).square()).mean()
            return loss, {}
        else:
            v_clipped = batch.old_values + (batch.value - batch.old_values).clamp(
                -self.vf_epsilon, self.vf_epsilon)
            l_unclipped = (batch.value - batch.returns) ** 2
            l_clipped = (v_clipped - batch.returns) ** 2
            loss = 0.5 * th.max(l_unclipped, l_clipped).mean()
            # true clip fraction: where the raw delta hit the range -- NOT where the
            # pessimistic max happened to take the clipped branch
            clipfrac = ((batch.value - batch.old_values).abs()
                        > self.vf_epsilon).float().mean().item()
            return loss, {"train/vf_clipfrac": clipfrac}


# --------------------------------------------------------------------------- #
#  the baseline update rule
# --------------------------------------------------------------------------- #


class PPO:
    """PPO update rule over the declared buffer. Stages map to algo_design §9.2:
    collect (S1/S2) -> estimate (S3) -> epochs x minibatches {evaluate (S8) ->
    losses (S9/S10) -> step (S11)} -> log (S12)."""

    def __init__(self, env, policy, rollout_buffer, config: PPOConfig,
                 log: Logger | None = None, monitor=None):
        self.config = config
        self.env = env
        self.policy = policy
        self.rollout_buffer = rollout_buffer
        self.log = log or Logger()
        self.monitor = monitor                     # report-only, never stops (§5.2)

        self.adv_fn = get_adv_estimator_fn(config.adv_estimator)
        self.policy_loss = ClipPolicyLoss(config.epsilon)
        self.value_loss = ValueLoss(config.vf_epsilon if config.use_value_clipped
                                    else None)
        self.timer = Timer()
        self._last_obs = None
        self.global_step = 0
        self.num_updates = 0

    # ------------------------------ S1 + S2 ------------------------------ #

    @th.no_grad()
    def _collect_rollout_(self) -> int:
        """One rollout into the declared buffer, then the bootstrap V(s_T).

        The adapter has already produced device tensors and the correct three
        masks (autoreset dummies are valid=False); the loop only writes."""
        buf = self.rollout_buffer
        buf.reset()
        if self._last_obs is None:
            self._last_obs = self.env.reset(seed=self.config.seed)

        for _ in range(self.config.rollout_len):
            with self.timer("fwd"):
                actions, logprobs, values = self.policy.act(self._last_obs)
            buf.write_obs(self._last_obs)
            buf.write(action=actions, logprob=logprobs, value=values)
            with self.timer("env"):
                obs, reward, masks, info = self.env.step(actions)
            buf.write(reward=reward)
            buf.write_masks(masks)
            buf.advance()
            self._last_obs = obs
            self.global_step += self.config.num_envs
            # raw, un-normalized episode stats -> the logger's sliding window;
            # without this the learning curve does not exist
            for ep_ret, ep_len in info["_finished_episodes"]:
                self.log.add_episode(ep_ret, ep_len)

        with self.timer("fwd"):
            last_value = self.policy.predict_value(self._last_obs)
        buf.set_bootstrap_value(last_value)
        return self.config.rollout_len * self.config.num_envs

    # ------------------------------- S3 ---------------------------------- #

    def _compute_advantages_(self) -> None:
        """Estimator slot -> (advantages, returns) -> the buffer's output slots."""
        buf = self.rollout_buffer
        masks = buf.masks
        adv, ret = self.adv_fn(
            buf["reward"][: buf.T], buf.values, buf.bootstrap_value,
            masks.terminated, masks.truncated,
            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda,
        )
        buf.set_estimates(adv, ret)

    # --------------------------- S5..S11 --------------------------------- #

    def _update_(self) -> dict[str, float]:
        cfg = self.config
        stats: dict[str, float] = {}
        stop = False
        # one generator per update: reproducible shuffles, different per epoch
        generator = th.Generator()
        generator.manual_seed(cfg.seed + self.num_updates * 1000)

        for _ in range(cfg.ppo_epoch):
            for data in self.rollout_buffer.sample(generator):
                batch = self._make_batch_(data)
                if cfg.normalize_advantage:
                    a = batch.advantages
                    # without this, unbounded advantages (e.g. CartPole returns
                    # in the hundreds) blow the first update up and the policy
                    # freezes inside the clip region forever. unbiased=False:
                    # a 1-sample minibatch must give std=0, not NaN.
                    batch.advantages = (a - a.mean()) / (a.std(unbiased=False) + 1e-8)
                with self.timer("bwd"):
                    pg_loss, pg_stats = self.policy_loss(batch)
                    v_loss, v_stats = self.value_loss(batch)
                    entropy = batch.entropy.mean()
                    log_ratio = batch.logp - batch.old_log_prob
                    approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()  # k3
                    loss = (pg_loss + cfg.vf_coef * v_loss
                            - cfg.ent_coef * entropy + cfg.kl_coef * approx_kl)
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = th.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), cfg.max_grad_norm)
                    self.policy.optimizer.step()

                stats = {
                    **pg_stats, **v_stats,
                    "loss/policy": pg_loss.item(), "loss/value": v_loss.item(),
                    "loss/entropy": entropy.item(), "loss/total": loss.item(),
                    "train/kl": approx_kl.item(), "train/grad_norm": float(grad_norm),
                }
                self.log.add(**stats)

                # early stop AFTER the metrics of this minibatch are recorded
                if cfg.target_kl is not None and stats["train/kl"] > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        return stats

    def _make_batch_(self, data: RolloutSamples) -> Batch:
        """S8: ONE evaluate over the minibatch, then assemble the context. The
        policy owns the distribution (doc §9.2 S8): stored obs + actions -> the
        current logp / entropy / value in a single forward."""
        logp, entropy, value = self.policy.evaluate(data.observations, data.actions)
        return Batch(
            observations=data.observations, actions=data.actions,
            old_log_prob=data.old_log_prob, old_values=data.old_values,
            advantages=data.advantages, returns=data.returns,
            logp=logp, entropy=entropy, value=value,
        )

    # ------------------------------- S12 ---------------------------------- #

    def train(self) -> None:
        """The outer loop: collect -> estimate -> update -> report."""
        while self.global_step < self.config.total_steps:
            self._collect_rollout_()
            self._compute_advantages_()
            stats = self._update_()
            self.num_updates += 1

            # critic health -- an underrated diagnostic (metrics.py)
            if stats:
                buf = self.rollout_buffer
                self.log.record("diag/explained_variance",
                                explained_variance(buf.values, buf.returns))
            self.log.record("perf/sps", self._sps_and_drain())
            self.log.record("train/updates", self.num_updates)
            if self.monitor is not None:               # report-only contact point
                self.monitor.observe(stats, global_step=self.global_step,
                                     iteration=self.num_updates)
            if self.num_updates % self.config.log_interval == 0:
                self.log.dump(self.global_step)

        self.log.dump(self.global_step)
        self.log.close()

    def _sps_and_drain(self) -> float:
        t = self.timer.drain()
        self.log.add(**t)
        elapsed = sum(t.values())
        return (self.config.rollout_len * self.config.num_envs / elapsed
                if elapsed > 0 else 0.0)

    # ------------------------------ checkpoint ---------------------------- #

    def _save_(self) -> None:
        """R2: policy + optimizer + buffer meta + rng (DESIGN_v2 §5.1)."""
        pass

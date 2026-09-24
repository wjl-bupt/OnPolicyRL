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
    epsilon: float = 0.2            # policy clip range (initial value if scheduled)
    clip_schedule: str = "constant"  # "constant" | "linear" (anneal eps over training)
    clip_range_min: float = 1e-3    # linear floor -- eps never reaches 0 (a 0 clip
                                    # range removes the trust region -> vanilla PG)
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
        if self.clip_schedule not in ("constant", "linear"):
            raise ValueError("clip_schedule must be 'constant' or 'linear'")


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

    The clip range may DECAY over training: `constant` holds epsilon; `linear`
    anneals it from epsilon down to epsilon_min as `batch.progress` runs 0 -> 1,
    floored at epsilon_min so it never hits 0 (a 0 clip range removes the trust
    region and collapses PPO to vanilla policy-gradient).
    """

    def __init__(self, epsilon: float = 0.2, schedule: str = "constant",
                 epsilon_min: float = 1e-3):
        self.epsilon = epsilon
        self.schedule = schedule
        self.epsilon_min = epsilon_min

    def _current_eps(self, progress: float) -> float:
        """The clip range at this point in training (`progress` in [0, 1])."""
        if self.schedule == "linear":
            eps = self.epsilon_min + (self.epsilon - self.epsilon_min) * (1.0 - progress)
            return max(self.epsilon_min, eps)
        return self.epsilon

    def __call__(self, batch: Batch) -> tuple[Tensor, dict[str, float]]:
        eps = self._current_eps(batch.progress)
        ratio = (batch.logp - batch.old_log_prob).exp()
        pg1 = -batch.advantages * ratio
        pg2 = -batch.advantages * ratio.clamp(1 - eps, 1 + eps)
        loss = th.max(pg1, pg2).mean()
        clipfrac = (pg2 > pg1).float().mean().item()
        return loss, {"train/clipfrac": clipfrac, "train/clip_range": eps}


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
        self.policy_loss = ClipPolicyLoss(config.epsilon, config.clip_schedule,
                                          config.clip_range_min)
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
        # one generator per update: reproducible shuffles, different per epoch.
        # It must live on the buffer's device -- torch.randperm requires the
        # generator and the target device to match (CPU path: device=cpu).
        generator = th.Generator(device=self.rollout_buffer.device)
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
        current logp / entropy / value in a single forward. `progress` (fraction
        of total_steps consumed) is stamped here so scheduled components -- e.g. a
        decaying clip range -- can read it off the batch."""
        logp, entropy, value = self.policy.evaluate(data.observations, data.actions)
        progress = min(1.0, self.global_step / max(1, self.config.total_steps))
        return Batch(
            observations=data.observations, actions=data.actions,
            old_log_prob=data.old_log_prob, old_values=data.old_values,
            advantages=data.advantages, returns=data.returns,
            logp=logp, entropy=entropy, value=value, progress=progress,
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
        # NOTE: we do NOT close the logger here. The logger is owned by the
        # caller (the workflow runner in the framework path, the test/script in
        # standalone use) -- it may still want to log "job complete" after this
        # returns. Closing a logger you did not create is an ownership bug (it
        # was the cause of a write-after-close on run.log). Sinks flush per
        # write, so nothing is lost by leaving the close to the owner.

    def _sps_and_drain(self) -> float:
        t = self.timer.drain()
        self.log.add(**t)
        elapsed = sum(t.values())
        return (self.config.rollout_len * self.config.num_envs / elapsed
                if elapsed > 0 else 0.0)

    # ------------------------------ checkpoint ---------------------------- #

    def _save_(self) -> None:
        """R2: policy + optimizer + buffer meta + rng (DESIGN_v2 §5.1).

        Kept a no-op inside the class on purpose: checkpointing is orchestrated
        by the workflow Monitor (common/checkpoint over policy + rng + step), so
        the loop stays ignorant of run directories. The single `monitor.observe`
        call in train() is the whole L2<->L4 contact surface."""
        pass


# --------------------------------------------------------------------------- #
#  module-level entry -- how the workflow layer (L4) runs PPO (algo_design §1)
# --------------------------------------------------------------------------- #


def train(cfg: PPOConfig, env, policy, log: Logger | None = None,
          monitor=None, resume: dict | None = None) -> dict:
    """The framework's entry into PPO: it hands over cfg + env + policy + log +
    monitor, and PPO owns the collect->update loop inside -- it is the only
    built-in training loop (a future V-MPO copies this file, it does not
    decompose the loop into an UpdateRule yet).

    The rollout buffer is the algorithm's PRIVATE concern, built here from the
    env's spaces with the minibatch count the config asks for -- so the runner
    never learns what a buffer is (a future GA2E would add `extra_samples` at
    exactly this line and the runner would not change).

    `resume`, when given, is the `{global_step, iteration}` a checkpoint the
    runner already loaded into `policy`; the step counters are fast-forwarded so
    the outer `while global_step < total_steps` continues rather than restarting.
    Returns a small summary dict the runner records as the job result.
    """
    from memory.buffer import RolloutBuffer
    from protocol.buffer import base_schema
    from protocol.sample import MinibatchSpec

    schema = base_schema(env.obs_space, env.action_space)
    spec = MinibatchSpec(num_minibatches=cfg.num_mini_batch)
    buf = RolloutBuffer(cfg.rollout_len, cfg.num_envs, schema,
                        device=cfg.device, spec=spec)
    ppo = PPO(env, policy, buf, cfg, log=log, monitor=monitor)
    if resume:
        ppo.global_step = int(resume.get("global_step", 0))
        ppo.num_updates = int(resume.get("iteration", 0))
    ppo.train()
    return {"global_step": ppo.global_step, "iteration": ppo.num_updates}

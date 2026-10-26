"""PPO with Partial GAE (Jin et al., IEEE Transactions on Games).

Use a T-step lookahead window, train its first K rows and completed episode
segments, and retain its unconsumed suffix. Behavior log probabilities never
change. An optional V-trace target branch corrects carried trajectories;
the default remains the original GAE branch. See doc/advantage_estimators.md.
"""
from dataclasses import dataclass
import math
import torch as th
from architecture.actor_critic import build_policy as build_base
from onpolicy.ppo import PPOConfig
from onpolicy._runtime import NativeLoop, estimates, make_buffer


@dataclass
class PAEConfig(PPOConfig):
    partial_steps: int | None = None  # None -> max(1, rollout_len // 2)
    use_vtrace: bool = False
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0
    vtrace_lambda: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.partial_steps is None:
            self.partial_steps = max(1, self.rollout_len // 2)
        if type(self.partial_steps) is not int or not 1 <= self.partial_steps <= self.rollout_len:
            raise ValueError('partial_steps must be an integer in [1, rollout_len]')
        if type(self.use_vtrace) is not bool:
            raise ValueError('use_vtrace must be boolean')
        if (not all(math.isfinite(x) and x > 0 for x in (self.vtrace_rho_clip, self.vtrace_c_clip))
                or self.vtrace_c_clip > self.vtrace_rho_clip):
            raise ValueError('V-trace requires 0 < c_clip <= rho_clip, both finite')
        if not math.isfinite(self.vtrace_lambda) or not 0 <= self.vtrace_lambda <= 1:
            raise ValueError('vtrace_lambda must lie in [0, 1]')


@th.no_grad()
def vtrace_targets(rewards, values, next_values, reference_logp, behavior_logp,
                   terminated, truncated, valid, gamma, rho_clip=1., c_clip=1., trace_lambda=1.):
    """[T,N] V-trace targets and UNWEIGHTED q-V for the existing PPO ratio.

    Recurse only along valid transitions in one episode. At rollout boundaries
    and time limits use next_values, at true terminals bootstrap zero. The
    current-step rho belongs to value correction, not an extra actor weight.
    """
    log_ratio = th.where(valid, reference_logp - behavior_logp, 0.)
    if not th.isfinite(log_ratio[valid]).all():
        raise FloatingPointError('Non-finite V-trace behavior/reference log probability')
    # Clamp in log space BEFORE exp so extreme stale-policy ratios cannot overflow.
    rho = th.where(valid, log_ratio.clamp_max(math.log(rho_clip)).exp(), 0.)
    c = th.where(valid, trace_lambda * log_ratio.clamp_max(math.log(c_clip)).exp(), 0.)
    bootstrap = (~terminated & valid).to(values.dtype)
    continuation = (~(terminated | truncated) & valid).clone()
    continuation[:-1] &= valid[1:]
    continuation[-1] = False
    td = th.where(valid, rewards + gamma * bootstrap * next_values - values, 0.)
    correction = th.zeros_like(values[0])
    targets = th.zeros_like(values)
    advantages = th.zeros_like(values)
    for t in reversed(range(len(values))):
        future = th.where(continuation[t], correction, 0.)
        advantages[t] = th.where(valid[t], td[t] + gamma * bootstrap[t] * future, 0.)
        correction = rho[t] * td[t] + gamma * c[t] * future
        targets[t] = th.where(valid[t], values[t] + correction, 0.)
    return advantages, targets, rho, c


class PAE(NativeLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._carry = None
        self._pending = None
        self._new_steps = 0
        # Checkpoints are made after update by the monitor; this callback is
        # optional and does not change the public Policy act/evaluate contract.
        self.policy.algorithm_state_dict = self.algorithm_state_dict

    @th.no_grad()
    def _collect_rollout_(self):
        b = self.rollout_buffer
        b.reset()
        self._pending = th.zeros((b.T, b.N), dtype=th.bool, device=b.device)
        self._had_carry = self._carry is not None
        start = 0
        if self._carry is not None:
            start = self._carry['pending'].shape[0]
            for key, value in self._carry['fields'].items():
                b[key][:start].copy_(value)
            self._pending[:start].copy_(self._carry['pending'])
            b.pos = start
        if self._last_obs is None:
            self._last_obs = self.env.reset(seed=self.config.seed)
        for t in range(start, b.T):
            with self.timer('fwd'):
                action, logp, value = self.policy.act(self._last_obs)
            b.write_obs(self._last_obs)
            b.write(action=action, logprob=logp, value=value)
            with self.timer('env'):
                obs, reward, masks, info = self.env.step(action)
            successor = info.get('_transition_obs', obs)
            b.write(next_obs=successor, next_value=self.policy.predict_value(successor), reward=reward)
            b.write_masks(masks)
            self._pending[t].copy_(masks.valid)
            b.advance()
            self._last_obs = obs
            self.global_step += b.N
            for ret, length in info['_finished_episodes']:
                self.log.add_episode(ret, length)
        b.set_bootstrap_value(self.policy.predict_value(self._last_obs))
        self._new_steps = (b.T - start) * b.N
        self.log.record('pae/new_env_steps', self._new_steps)
        self.log.record('pae/carried_pending', 0 if self._carry is None else int(self._carry['pending'].sum()))
        return self._new_steps

    @th.no_grad()
    def _compute_advantages_(self):
        b, c = self.rollout_buffer, self.config
        if c.use_vtrace:
            # Reference policy is the current policy BEFORE any epoch updates.
            # Targets stay fixed throughout this update; behavior logp is untouched.
            reference_logp, _, values = self.policy.evaluate(self.flat('obs'), self.flat('action'))
            next_values = self.policy.predict_value(self.flat('next_obs')).reshape(b.T, b.N)
            valid = b['valid'][:b.T].bool()
            adv, returns, rho, trace = vtrace_targets(
                b['reward'][:b.T], values.reshape(b.T, b.N), next_values,
                reference_logp.reshape(b.T, b.N), b['logprob'][:b.T],
                b['terminated'][:b.T], b['truncated'][:b.T], valid,
                c.gamma, c.vtrace_rho_clip, c.vtrace_c_clip, c.vtrace_lambda)
            if valid.any():
                log_ratio = reference_logp.reshape(b.T, b.N)[valid] - b['logprob'][:b.T][valid]
                self.log.record('pae/vtrace_rho_mean', float(rho[valid].mean()))
                self.log.record('pae/vtrace_c_mean', float(trace[valid].mean()))
                self.log.record('pae/vtrace_rho_clipfrac', float((log_ratio > math.log(c.vtrace_rho_clip)).float().mean()))
        elif self._had_carry:
            # Refresh target values on the combined window. Preserve sampled
            # value/logprob fields for PPO value clipping and behavior ratios.
            values = self.policy.predict_value(self.flat('obs')).reshape(b.T, b.N)
            next_values = self.policy.predict_value(self.flat('next_obs')).reshape(b.T, b.N)
            adv, returns = estimates(b, c.gamma, c.gae_lambda, values, next_values)
        else:
            adv, returns = estimates(b, c.gamma, c.gae_lambda)
        b.set_estimates(adv, returns)
        valid = b['valid'][:b.T].bool()
        closed = (b['terminated'][:b.T] | b['truncated'][:b.T]) & valid
        # A finished segment has no future transitions to wait for; time limits
        # keep their terminal-observation bootstrap in estimates().
        finished_suffix = closed.flip(0).long().cumsum(0).flip(0) > 0
        prefix = th.arange(b.T, device=b.device)[:, None] < c.partial_steps
        self._eligible = valid & self._pending & (prefix | finished_suffix)
        self.log.record('pae/train_samples', int(self._eligible.sum()))

    def _update_(self):
        b = self.rollout_buffer
        valid = b['valid'][:b.T].clone()
        # Use one common sample mask for actor, critic, entropy and diagnostics;
        # restore the trajectory-valid mask before preserving the suffix.
        try:
            b['valid'][:b.T].copy_(self._eligible)
            stats = super()._update_()
        finally:
            b['valid'][:b.T].copy_(valid)
        self._pending &= ~self._eligible
        start = self.config.partial_steps
        if start < b.T and self._pending[start:].any():
            self._carry = {
                'fields': {key: b[key][start:b.T].detach().clone() for key in b.schema},
                'pending': self._pending[start:].clone(),
            }
        else:
            self._carry = None
        self.log.record('pae/pending_tail', int(self._pending.sum()))
        return stats

    def _sps_and_drain(self):
        times = self.timer.drain()
        self.log.add(**times)
        elapsed = sum(times.values())
        return self._new_steps / elapsed if elapsed > 0 else 0.

    def algorithm_state_dict(self):
        carry = None if self._carry is None else {
            'fields': {k: v.detach().cpu().clone() for k, v in self._carry['fields'].items()},
            'pending': self._carry['pending'].cpu().clone(),
        }
        return dict(version=1, partial_steps=self.config.partial_steps,
                    rollout_len=self.config.rollout_len, carry=carry,
                    last_obs=None if self._last_obs is None else self._last_obs.detach().cpu().clone())

    def load_algorithm_state(self, state, *, trajectory_restored=False):
        if not state:
            return
        if (state.get('version') != 1 or state['partial_steps'] != self.config.partial_steps
                or state['rollout_len'] != self.config.rollout_len):
            raise ValueError('Incompatible PAE carry checkpoint')
        carry = state['carry']
        if not trajectory_restored:
            count = 0 if carry is None else int(carry['pending'].sum())
            self.log.record('pae/resume_discarded_pending', count)
            if count:
                self.log.warning('PAE resume: simulator state was not restored; discarding pending tail', samples=count)
            self._carry = None
            self._last_obs = None
            return
        device = self.rollout_buffer.device
        self._carry = None if carry is None else {
            'fields': {k: v.to(device).clone() for k, v in carry['fields'].items()},
            'pending': carry['pending'].to(device).clone(),
        }
        self._last_obs = None if state['last_obs'] is None else state['last_obs'].to(device).clone()


CONFIG_CLASS = PAEConfig


def build_policy(obs, action, network, algo):
    if network.action_head == 'diffusion':
        raise ValueError('PAE requires a Categorical or Gaussian policy, not diffusion')
    return build_base(obs, action, network)


def train(cfg, env, policy, log=None, monitor=None, resume=None):
    algo = PAE(env, policy, make_buffer(cfg, env), cfg, log=log, monitor=monitor)
    if resume:
        algo.global_step = int(resume['global_step'])
        algo.num_updates = int(resume['iteration'])
        # Current adapters restore normalization statistics, not simulator/RNG
        # state. Only an adapter explicitly guaranteeing exact restoration may
        # attach the saved suffix to freshly collected data.
        algo.load_algorithm_state(resume.get('algorithm_state'),
            trajectory_restored=bool(getattr(env, 'supports_exact_resume', False)))
    algo.train()
    return {'global_step': algo.global_step, 'iteration': algo.num_updates}

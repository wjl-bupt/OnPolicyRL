"""Native rollout utilities; no external RL framework owns any training state."""
import torch as th
from onpolicy.ppo import PPO
from memory.buffer import RolloutBuffer
from protocol.buffer import base_schema, Field
from protocol.sample import MinibatchSpec


def scan(deltas, continuation, discount):
    carry = th.zeros_like(deltas[0])
    result = []
    for t in reversed(range(len(deltas))):
        carry = deltas[t] + discount * continuation[t] * carry
        result.append(carry)
    return th.stack(result[::-1])


def estimates(buf, gamma, lam, values=None, next_values=None):
    values = buf.values if values is None else values
    next_values = buf['next_value'] if next_values is None else next_values
    m = buf.masks
    delta = buf['reward'] + gamma * (~m.terminated) * next_values - values
    cont = (~(m.terminated | m.truncated) & m.valid).float()
    adv = scan(delta, cont, gamma * lam) * m.valid
    return adv.detach(), (adv + values).detach()


def make_buffer(cfg, env):
    if cfg.num_envs != env.num_envs:
        raise ValueError("num_envs must match the environment adapter")
    schema = base_schema(env.obs_space, env.action_space)
    if 'obs' not in schema:
        raise NotImplementedError('Native algorithms currently require numeric vector/image observations')
    schema.update(next_obs=Field(schema['obs'].shape, schema['obs'].dtype), next_value=Field())
    return RolloutBuffer(cfg.rollout_len, cfg.num_envs, schema, cfg.device,
                         MinibatchSpec(num_minibatches=cfg.num_mini_batch))


class NativeLoop(PPO):
    """Shared native lifecycle, with algorithm-specific estimation and update methods."""
    @th.no_grad()
    def _collect_rollout_(self):
        buf = self.rollout_buffer
        buf.reset()
        if self._last_obs is None: self._last_obs = self.env.reset(seed=self.config.seed)
        for _ in range(buf.T):
            with self.timer('fwd'):
                actions, logp, values = self.policy.act(self._last_obs)
            buf.write_obs(self._last_obs)
            buf.write(action=actions, logprob=logp, value=values)
            with self.timer('env'):
                obs, reward, masks, info = self.env.step(actions)
            successor = info.get('_transition_obs', obs)
            buf.write(next_obs=successor, next_value=self.policy.predict_value(successor), reward=reward)
            buf.write_masks(masks)
            buf.advance()
            self._last_obs = obs
            self.global_step += buf.N
            for ret, length in info['_finished_episodes']: self.log.add_episode(ret, length)
        buf.set_bootstrap_value(self.policy.predict_value(self._last_obs))
        return buf.T * buf.N

    def _compute_advantages_(self):
        self.rollout_buffer.set_estimates(*estimates(self.rollout_buffer, self.config.gamma, self.config.gae_lambda))

    def step_loss(self, loss):
        if not th.isfinite(loss): raise FloatingPointError('Non-finite algorithm loss')
        opt = self.policy.optimizer
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        if not th.isfinite(norm): raise FloatingPointError('Non-finite algorithm gradient')
        opt.step()
        return float(norm)

    def flat(self, name):
        b = self.rollout_buffer
        if name == 'advantages': return b.advantages.flatten()
        if name == 'returns': return b.returns.flatten()
        return b[name][:b.T].flatten(0, 1)

    def indices(self):
        return self.flat('valid').bool().nonzero().flatten()

    def batches(self):
        idx = self.indices()
        gen = th.Generator(device=idx.device).manual_seed(self.config.seed + self.num_updates * 1000)
        for _ in range(self.config.ppo_epoch):
            yield from (part for part in th.tensor_split(idx[th.randperm(len(idx),device=idx.device,generator=gen)], self.config.num_mini_batch) if len(part))


def run_algorithm(cls, cfg, env, policy, log=None, monitor=None, resume=None):
    algo = cls(env, policy, make_buffer(cfg,env),cfg,log=log,monitor=monitor)
    if resume:
        algo.global_step = int(resume['global_step'])
        algo.num_updates = int(resume['iteration'])
    algo.train()
    return {'global_step':algo.global_step,'iteration':algo.num_updates}

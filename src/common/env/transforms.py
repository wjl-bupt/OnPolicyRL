"""Identical post-simulation transforms for both tensor adapters.

Running statistics are checkpointed by the runner and frozen during evaluation.
Simulation remains NumPy/JAX; these transforms execute on the model device.
"""
import numpy as np
import torch
from gymnasium import spaces

DEFAULTS = dict(normalize_observation=False, observation_clip=None,
                normalize_reward=False, reward_clip=None, reward_sign=False,
                gamma=.99, epsilon=1e-8, action_mode='clip', frame_stack=1)


def validate_transforms(options):
    if not isinstance(options, dict) or set(options) - DEFAULTS.keys():
        raise ValueError(f'unknown wrapper options: {options}')
    cfg = {**DEFAULTS, **options}
    for k in ('normalize_observation', 'normalize_reward', 'reward_sign'):
        if type(cfg[k]) is not bool: raise ValueError(f'{k} must be boolean')
    if cfg['action_mode'] not in ('clip', 'rescale', 'none'): raise ValueError('invalid action_mode')
    if type(cfg['frame_stack']) is not int or cfg['frame_stack'] < 1: raise ValueError('invalid frame_stack')
    for k in ('observation_clip', 'reward_clip', 'epsilon'):
        if cfg[k] is not None and (not np.isfinite(cfg[k]) or cfg[k] <= 0): raise ValueError(f'invalid {k}')
    if not 0 <= cfg['gamma'] <= 1: raise ValueError('invalid normalization gamma')
    return cfg


class Moments:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.var = torch.ones_like(self.mean)
        self.count = 0.

    def update(self, x):
        if len(x) == 0: return
        x = x.double()
        n, mean, var = len(x), x.mean(0), x.var(0, unbiased=False)
        delta, total = mean - self.mean, self.count + n
        self.var = (self.var * self.count + var * n + delta.square() * self.count * n / total) / total
        self.mean += delta * n / total
        self.count = float(total)

    def state_dict(self):
        return dict(mean=self.mean.cpu().clone(), var=self.var.cpu().clone(), count=self.count)

    def load_state_dict(self, state):
        if state['mean'].shape != self.mean.shape: raise ValueError('normalizer shape mismatch')
        self.mean.copy_(state['mean']); self.var.copy_(state['var']); self.count = float(state['count'])


class TensorTransforms:
    def __init__(self, env, options=None, *, training=True):
        self.env, self.training = env, training
        self.options = validate_transforms(options or {})
        self.device, self.num_envs = env.device, env.num_envs
        self.action_space, self.obs_space = env.action_space, env.obs_space
        if not isinstance(self.obs_space, spaces.Box): raise ValueError('encode dictionary observations before tensor transforms')
        self.obs_moments = Moments(self.obs_space.shape, self.device)
        self.reward_moments = Moments((), self.device)
        self.returns = torch.zeros(self.num_envs, device=self.device)
        self.stack = self.options['frame_stack']
        self.history = None
        if self.options['normalize_observation']:
            bound = self.options['observation_clip'] or np.inf
            self.obs_space = spaces.Box(-bound, bound, self.obs_space.shape, dtype=np.float32)
        if self.stack > 1:
            self.obs_space = spaces.Box(np.concatenate([self.obs_space.low]*self.stack, 0),
                                       np.concatenate([self.obs_space.high]*self.stack, 0), dtype=self.obs_space.dtype)
        if isinstance(self.action_space, spaces.Box):
            self.low = torch.as_tensor(self.action_space.low, device=self.device)
            self.high = torch.as_tensor(self.action_space.high, device=self.device)
            if self.options['action_mode'] == 'rescale':
                if not torch.isfinite(self.low).all() or not torch.isfinite(self.high).all(): raise ValueError('rescale needs finite bounds')
                self.action_space = spaces.Box(-1., 1., self.action_space.shape, dtype=np.float32)
        self.provenance = dict(getattr(env, 'provenance', {}), wrappers=self.options, wrapper_engine='torch-v1')

    def _obs(self, obs):
        if not self.options['normalize_observation']: return obs
        obs = (obs.float() - self.obs_moments.mean.float()) / torch.sqrt(self.obs_moments.var.float() + self.options['epsilon'])
        clip = self.options['observation_clip']
        return obs.clamp(-clip, clip) if clip else obs

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        self.returns.zero_()
        if self.training and self.options['normalize_observation']: self.obs_moments.update(obs)
        obs = self._obs(obs)
        self.history = [obs.clone() for _ in range(self.stack)]
        return torch.cat(self.history, 1) if self.stack > 1 else obs

    def step(self, actions):
        if isinstance(self.action_space, spaces.Box):
            if self.options['action_mode'] == 'rescale': actions = self.low + (actions.clamp(-1, 1)+1)*.5*(self.high-self.low)
            elif self.options['action_mode'] == 'clip': actions = actions.clamp(self.low, self.high)
        obs, reward, masks, info = self.env.step(actions)
        info = dict(info)
        raw = reward.clone()
        info['_raw_reward'] = raw
        final = info['_transition_obs']
        if self.training and self.options['normalize_observation']: self.obs_moments.update(final[masks.valid])
        obs, final = self._obs(obs), self._obs(final)
        done = masks.terminated | masks.truncated
        if self.stack > 1:
            previous = self.history[1:]
            info['_transition_obs'] = torch.cat(previous + [final], 1)
            reset = ~masks.valid if getattr(self.env, 'next_step_autoreset', False) else done
            self.history = [torch.where(reset.reshape((-1,)+(1,)*(obs.ndim-1)), obs, h) for h in previous+[obs]]
            obs = torch.cat(self.history, 1)
        else: info['_transition_obs'] = final
        if self.options['normalize_reward']:
            self.returns = torch.where(masks.valid, self.returns*self.options['gamma'] + raw, self.returns)
            if self.training:
                self.reward_moments.update(self.returns[masks.valid])
                reward = raw / torch.sqrt(self.reward_moments.var.float()+self.options['epsilon'])
            self.returns[done] = 0
        if self.training and self.options['reward_sign']: reward = reward.sign()
        if self.training and self.options['reward_clip']: reward = reward.clamp(-self.options['reward_clip'], self.options['reward_clip'])
        return obs, reward, masks, info

    def state_dict(self):
        return dict(version=1, options=dict(self.options), observation=self.obs_moments.state_dict(), reward=self.reward_moments.state_dict())

    def load_state_dict(self, state):
        if state['options'] != self.options: raise ValueError('wrapper configuration mismatch')
        self.obs_moments.load_state_dict(state['observation']); self.reward_moments.load_state_dict(state['reward'])

    def close(self): self.env.close()

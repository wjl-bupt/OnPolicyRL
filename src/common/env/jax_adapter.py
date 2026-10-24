"""JIT/vmap environment backends implementing OnPolicyRL's tensor protocol.

Gymnax supplies MinAtar dynamics; Brax supplies tasks using MuJoCo's MJX
physics. Policies, rollout buffers, estimators and updates remain native torch.
Natural termination and time limits stay separate; same-step resets retain the
pre-reset successor for value bootstrapping. No upstream training code is used.
"""
from functools import lru_cache
from typing import NamedTuple, Any
from common.env.minatar_options import sticky_probability
import numpy as np
import torch
from gymnasium import spaces
from protocol.masks import Masks


class _StickyState(NamedTuple):
    env_state: Any
    last_action: Any


@lru_cache(maxsize=32)
def _minatar_kernel(name, minimal, limit, sticky):
    import gymnax
    import jax
    import jax.numpy as jp
    env, params = gymnax.make(name, use_minimal_action_set=minimal)
    params = params.replace(max_steps_in_episode=limit)

    def reset(key):
        obs, state = env.reset(key, params)
        return obs, _StickyState(state, jp.array(0, dtype=jp.int32))

    def raw_step(key, state, action):
        if sticky == 0.0:
            executed = action
        else:
            replay_key, key = jax.random.split(key)
            executed = jp.where(jax.random.bernoulli(replay_key, sticky), state.last_action, action)
        obs, inner, reward, term, _ = env.step_env(key, state.env_state, executed, params)
        return obs, _StickyState(inner, executed), reward, term, env.is_truncated(inner, params)

    obs_space = env.observation_space(params)
    shape = (obs_space.shape[2], obs_space.shape[0], obs_space.shape[1])
    return reset, raw_step, spaces.Box(0, 1, shape, dtype=np.float32), spaces.Discrete(env.num_actions)


@lru_cache(maxsize=32)
def _mjx_kernel(name, options):
    from brax import envs
    import jax.numpy as jp
    env = envs.get_environment(name, backend='mjx', **dict(options))

    def reset(key):
        state = env.reset(key)
        return state.obs, state

    def step(key, state, action):
        del key
        state = env.step(state, jp.clip(action, -1, 1))
        return state.obs, state, state.reward, state.done.astype(bool), jp.array(False)

    return (reset, step, spaces.Box(-np.inf, np.inf, (env.observation_size,), dtype=np.float32),
            spaces.Box(-1, 1, (env.action_size,), dtype=np.float32))


@lru_cache(maxsize=64)
def _batch_kernels(kind, name, minimal, limit, options, sticky):
    import jax
    import jax.numpy as jp
    reset, step, obs_space, action_space = (_minatar_kernel(name, minimal, limit, sticky)
        if kind == 'minatar' else _mjx_kernel(name, options))

    def one(key, state, action, ep_return, ep_length):
        next_key, step_key, reset_key = jax.random.split(key, 3)
        final_obs, successor, reward, term, trunc = step(step_key, state, action)
        length = ep_length + 1
        trunc = trunc | (length >= limit)
        done = term | trunc
        total = ep_return + reward
        # vmap batches both reset branches; select fresh states only when done.
        obs, state = jax.lax.cond(done, lambda: reset(reset_key), lambda: (final_obs, successor))
        if kind == 'minatar':
            # Match original MinAtar: automatic reset preserves the last
            # executed action; an explicit adapter.reset starts from action 0.
            state = state._replace(last_action=successor.last_action)
        return (next_key, state, obs, final_obs, reward, term, trunc,
                jp.where(done, 0., total), jp.where(done, 0, length), total, length)

    return jax.jit(jax.vmap(reset)), jax.jit(jax.vmap(one)), obs_space, action_space


class JaxVectorAdapter:
    """Explicit JAX simulation device, independent from the torch model device."""
    def __init__(self, env_id, num_envs=1, seed=0, device='cpu', *,
                 jax_platform='cpu', jax_device_index=0, use_minimal_action_set=False,
                 max_episode_steps=None, autoreset_mode='SameStep', **task_options):
        import jax
        import jax.numpy as jp
        if autoreset_mode != 'SameStep':
            raise ValueError('JAX adapters implement SameStep resets only')
        if num_envs < 1:
            raise ValueError('num_envs must be positive')
        if env_id.startswith('MinAtar/'):
            kind = 'minatar'
            sticky = sticky_probability(task_options.pop('sticky_action_prob', 0.1))
            game = env_id.split('/')[1].split('-')[0]
            name = game + '-MinAtar'
            if game not in ('Asterix', 'Breakout', 'Freeway', 'Seaquest', 'SpaceInvaders'):
                raise ValueError(f'Unsupported JAX MinAtar: {env_id}')
            if task_options:
                raise ValueError(f'Unknown JAX MinAtar options: {sorted(task_options)}')
            limit = 2500 if game == 'Freeway' else 1000
        elif env_id.startswith('MJX/'):
            kind, name, limit = 'mjx', env_id.split('/', 1)[1], 1000
            sticky = None
            if 'sticky_action_prob' in task_options:
                raise ValueError('sticky_action_prob is a MinAtar-only option')
        else:
            raise ValueError('Use MinAtar/<Game>-v0 or MJX/<brax_task>; MJX tasks are not Gymnasium v5 tasks')
        limit = limit if max_episode_steps is None else int(max_episode_steps)
        if limit < 1:
            raise ValueError('max_episode_steps must be positive')
        self.kind, self.num_envs, self.seed = kind, int(num_envs), int(seed or 0)
        self.device = torch.device(device)
        self.jax_device = jax.devices(jax_platform)[jax_device_index]
        with jax.default_device(self.jax_device):
            self._reset, self._step, self.obs_space, self.action_space = _batch_kernels(
                kind, name, bool(use_minimal_action_set), limit, tuple(sorted(task_options.items())), sticky)
        self._state = None
        self.provenance = dict(backend='gymnax' if kind=='minatar' else 'brax-mjx',
                               task=name, jax_platform=self.jax_device.platform,
                               max_episode_steps=limit, minimal_actions=bool(use_minimal_action_set))
        if kind == 'minatar':
            self.provenance['sticky_action_prob'] = sticky

    def _tensor(self, value):
        # DLPack transfers ownership correctly and avoids a numpy host roundtrip
        # when both engines run on the same accelerator. Never mutate these views.
        return torch.utils.dlpack.from_dlpack(value).to(self.device)

    def _obs(self, value):
        t = self._tensor(value)
        return t.permute(0, 3, 1, 2).contiguous().float() if self.kind == 'minatar' else t.float()

    def reset(self, seed=None):
        import jax
        import jax.numpy as jp
        with jax.default_device(self.jax_device):
            key = jax.random.PRNGKey(self.seed if seed is None else int(seed))
            keys = jax.random.split(key, self.num_envs * 2)
            self._keys = keys[:self.num_envs]
            obs, self._state = self._reset(keys[self.num_envs:])
            self._returns = jp.zeros(self.num_envs)
            self._lengths = jp.zeros(self.num_envs, dtype=jp.int32)
        return self._obs(obs)

    def step(self, actions):
        import jax
        import jax.numpy as jp
        if self._state is None:
            raise RuntimeError('reset() must precede step()')
        with jax.default_device(self.jax_device):
            a = actions.detach().contiguous()
            if self.jax_device.platform == 'cpu':
                a = jp.asarray(a.cpu().numpy())
            else:
                a = jax.dlpack.from_dlpack(a)
                a = jax.device_put(a, self.jax_device)
            if isinstance(self.action_space, spaces.Discrete): a = a.astype(jp.int32)
            (self._keys, self._state, obs, successor, reward, term, trunc,
             self._returns, self._lengths, totals, lengths) = self._step(
                 self._keys, self._state, a, self._returns, self._lengths)
            stats = np.asarray(jax.device_get(jp.stack([term | trunc, totals, lengths], axis=1)))
        term_t, trunc_t = self._tensor(term).bool(), self._tensor(trunc).bool()
        info = {'_transition_obs': self._obs(successor),
                '_finished_episodes': [(float(r), int(l)) for done, r, l in stats if done],
                '_finished_episode_indices': np.flatnonzero(stats[:, 0]).tolist()}
        if self.kind == 'minatar':
            info['_executed_actions'] = self._tensor(self._state.last_action)
        return self._obs(obs), self._tensor(reward).float(), Masks(term_t, trunc_t, torch.ones_like(term_t)), info

    def close(self):
        self._state = None

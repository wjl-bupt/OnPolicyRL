"""Optional JAX providers with one reset/termination/vectorization protocol."""
import numpy as np
import torch
from gymnasium import spaces
from protocol.masks import Masks


def provider(env_id, options):
    import jax
    import jax.numpy as jp
    options = dict(options)
    if env_id.startswith('Craftax'):
        from craftax.craftax_env import make_craftax_env_from_name
        env = make_craftax_env_from_name(env_id, auto_reset=False)
        params = env.default_params.replace(**options)
        def reset(key): return env.reset(key, params)
        def step(key, state, action):
            obs, state, reward, done, info = env.step_env(key, state, action, params)
            trunc = state.timestep >= params.max_timesteps
            # Classify the natural terminal separately, even at the time limit.
            term = env.is_terminal(state, params.replace(max_timesteps=2**30))
            return obs, state, reward, term, trunc, {'achievements': state.achievements}
        s = env.observation_space(params)
        pixels = 'Pixels' in env_id
        shape = (s.shape[2], s.shape[0], s.shape[1]) if pixels else s.shape
        def encode(obs): return jp.transpose(obs, (2,0,1)).astype(jp.float32) if pixels else obs.astype(jp.float32)
        raw_reset, raw_step = reset, step
        def reset(key):
            obs, state = raw_reset(key); return encode(obs), state
        def step(key, state, action):
            obs, state, r, t, tr, info = raw_step(key, state, action)
            return encode(obs), state, r, t, tr, info
        return reset, step, spaces.Box(-np.inf, np.inf, shape, dtype=np.float32), spaces.Discrete(env.num_actions), 'craftax'
    if env_id.startswith('XMiniGrid/'):
        import xminigrid
        from xminigrid.core.constants import NUM_TILES, NUM_COLORS
        env, params = xminigrid.make(env_id.split('/',1)[1])
        params = params.replace(**options)
        def encode(ts):
            obs = ts.observation
            return jp.concatenate([jax.nn.one_hot(obs[...,0], NUM_TILES).ravel(),
                                   jax.nn.one_hot(obs[...,1], NUM_COLORS).ravel(),
                                   jax.nn.one_hot(ts.state.agent.direction, 4)])
        def reset(key):
            ts = env.reset(params, key); return encode(ts), ts
        def step(key, ts, action):
            ts = env.step(params, ts, action)
            return encode(ts), ts, ts.reward, ts.last() & (ts.discount == 0), ts.last() & (ts.discount != 0), {}
        n = params.view_size**2*(NUM_TILES+NUM_COLORS)+4
        return reset, step, spaces.Box(0,1,(n,),dtype=np.float32), spaces.Discrete(env.num_actions(params)), 'xminigrid'
    if env_id.startswith('Playground/'):
        from mujoco_playground import dm_control_suite
        options.setdefault('impl', 'jax')
        env = dm_control_suite.load(env_id.split('/',1)[1], config_overrides=options)
        def encode(obs):
            if isinstance(obs, dict): return jp.concatenate([obs[k].ravel() for k in sorted(obs)])
            return obs
        def reset(key):
            state = env.reset(key); return encode(state.obs), state
        def step(key, state, action):
            state = env.step(state, action)
            return encode(state.obs), state, state.reward, state.done.astype(bool), jp.array(False), {}
        example = jax.eval_shape(reset, jax.random.PRNGKey(0))[0]
        return reset, step, spaces.Box(-np.inf,np.inf,example.shape,dtype=np.float32), spaces.Box(-1,1,(env.action_size,),dtype=np.float32), 'playground-mjx'
    if env_id.startswith('JAXAtari/'):
        from common.env.jax_atari import atari_provider
        return atari_provider(env_id.split('/',1)[1], options)
    import gymnax
    env, params = gymnax.make(env_id)
    params = params.replace(**options)
    def reset(key): return env.reset(key, params)
    def step(key, state, action):
        obs, state, r, term, info = env.step_env(key, state, action, params)
        trunc = env.is_truncated(state, params) if hasattr(env,'is_truncated') else state.time >= params.max_steps_in_episode
        return obs, state, r, term & ~trunc, trunc, {}
    s, a = env.observation_space(params), env.action_space(params)
    action = spaces.Discrete(a.n) if hasattr(a,'n') else spaces.Box(np.asarray(a.low),np.asarray(a.high),shape=a.shape,dtype=np.float32)
    return reset, step, spaces.Box(-np.inf,np.inf,s.shape,dtype=np.float32), action, 'gymnax'


class FunctionalAdapter:
    def __init__(self, env_id, num_envs=1, seed=0, device='cpu', *, jax_platform='cpu',
                 jax_device_index=0, max_episode_steps=1000, action_repeat=1,
                 autoreset_mode='SameStep', **options):
        import jax
        import jax.numpy as jp
        if autoreset_mode != 'SameStep': raise ValueError('functional adapters require SameStep')
        if type(action_repeat) is not int or action_repeat < 1: raise ValueError('invalid action_repeat')
        if num_envs < 1 or max_episode_steps < 1: raise ValueError('positive num_envs and max_episode_steps required')
        self.num_envs, self.seed, self.device = num_envs, seed or 0, torch.device(device)
        self.jax_device = jax.devices(jax_platform)[jax_device_index]
        with jax.default_device(self.jax_device):
            reset, step, self.obs_space, self.action_space, implementation = provider(env_id, options)
        limit = int(max_episode_steps)
        def one(key, state, action, ret, length):
            key, sk, rk = jax.random.split(key,3)
            obs, state, reward, term, trunc, info = step(sk, state, action)
            for _ in range(action_repeat-1):
                key, sk = jax.random.split(key)
                def advance():
                    o,s,r,t,tr,i = step(sk,state,action)
                    return o,s,reward+r,t,tr,i
                obs,state,reward,term,trunc,info = jax.lax.cond(term|trunc,
                    lambda:(obs,state,reward,term,trunc,info), advance)
            length += 1
            trunc = trunc | (length >= limit)
            done, total = term | trunc, ret + reward
            final = obs
            obs,state = jax.lax.cond(done, lambda:reset(rk), lambda:(obs,state))
            return key,state,obs,final,reward,term,trunc,jp.where(done,0.,total),jp.where(done,0,length),total,length,info
        self._reset, self._step = jax.jit(jax.vmap(reset)), jax.jit(jax.vmap(one))
        self.provenance = dict(task=env_id, implementation=implementation, simulation=jax_platform,
                               max_episode_steps=limit, action_repeat=action_repeat, options=options)

    def _tensor(self, x): return torch.utils.dlpack.from_dlpack(x).to(self.device)

    def reset(self, seed=None):
        import jax
        import jax.numpy as jp
        with jax.default_device(self.jax_device):
            keys = jax.random.split(jax.random.PRNGKey(self.seed if seed is None else seed), self.num_envs*2)
            self.keys = keys[:self.num_envs]
            obs,self.state = self._reset(keys[self.num_envs:])
            self.returns, self.lengths = jp.zeros(self.num_envs), jp.zeros(self.num_envs,dtype=jp.int32)
        return self._tensor(obs)

    def step(self, actions):
        import jax
        import jax.numpy as jp
        with jax.default_device(self.jax_device):
            a = jp.asarray(actions.detach().cpu().numpy())
            if isinstance(self.action_space,spaces.Discrete): a=a.astype(jp.int32)
            (self.keys,self.state,obs,final,reward,term,trunc,self.returns,self.lengths,total,length,extra) = self._step(self.keys,self.state,a,self.returns,self.lengths)
            done = np.asarray(jax.device_get(term|trunc))
            totals,lengths = np.asarray(total),np.asarray(length)
        t,tr = self._tensor(term).bool(),self._tensor(trunc).bool()
        info = dict(_transition_obs=self._tensor(final),_finished_episode_indices=np.flatnonzero(done).tolist(),
                    _finished_episodes=[(float(totals[i]),int(lengths[i])) for i in np.flatnonzero(done)])
        if 'achievements' in extra:
            info['_finished_achievements'] = np.asarray(extra['achievements'])[done].tolist()
        return self._tensor(obs),self._tensor(reward).float(),Masks(t,tr,torch.ones_like(t)),info

    def close(self): pass

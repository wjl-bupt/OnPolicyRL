"""Gymnasium interface to Craftax; does not pretend to be a NumPy rewrite."""
import gymnasium as gym
import jax
import numpy as np
from common.env.functional import provider
from common.env.wrappers import vectorize


class CraftaxGym(gym.Env):
    def __init__(self, env_id, **options):
        self._reset,self._step,self.observation_space,self.action_space,_ = provider(env_id,options)
        self._reset,self._step = jax.jit(self._reset),jax.jit(self._step)
        self.key = jax.random.PRNGKey(0)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.key = jax.random.PRNGKey(seed)
        self.key,k = jax.random.split(self.key)
        obs,self.state = self._reset(k)
        return np.asarray(obs),{}
    def step(self, action):
        self.key,k = jax.random.split(self.key)
        obs,self.state,r,t,tr,info = self._step(k,self.state,action)
        return np.asarray(obs),float(r),bool(t),bool(tr),{k:np.asarray(v) for k,v in info.items()}


def make_craftax_gym(env_id,num_envs,seed,**options):
    mode = options.pop('autoreset_mode','SameStep')
    limit = options.pop('max_episode_steps',1000)
    repeat = options.pop('action_repeat',1)
    def one():
        from common.env.observations import ActionRepeat
        env = ActionRepeat(CraftaxGym(env_id,**options),repeat)
        return gym.wrappers.RecordEpisodeStatistics(gym.wrappers.TimeLimit(env,limit))
    return vectorize(one,num_envs,seed,False,mode)

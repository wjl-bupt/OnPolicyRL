"""dm_env -> Gymnasium without confusing time limits with termination."""
import gymnasium as gym
import numpy as np


class DMControlEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}
    def __init__(self, env_id, obs_mode='state', screen_size=84, camera_id=0, action_repeat=1):
        self.domain, self.task = env_id.split('/')[1:]
        self.obs_mode, self.screen_size, self.camera_id = obs_mode, screen_size, camera_id
        if obs_mode not in ('state','pixels'): raise ValueError('DMControl obs_mode must be state or pixels')
        if type(action_repeat) is not int or action_repeat < 1: raise ValueError('invalid action_repeat')
        self.action_repeat = action_repeat
        self._create(0)
        a = self.env.action_spec()
        self.action_space = gym.spaces.Box(a.minimum.astype(np.float32),a.maximum.astype(np.float32),dtype=np.float32)
        specs = self.env.observation_spec()
        self.keys = sorted(specs)
        shape = (3,screen_size,screen_size) if obs_mode=='pixels' else (sum(int(np.prod(specs[k].shape)) for k in self.keys),)
        self.observation_space = gym.spaces.Box(0,255,shape,dtype=np.uint8) if obs_mode=='pixels' else gym.spaces.Box(-np.inf,np.inf,shape,dtype=np.float32)

    def _create(self, seed):
        from dm_control import suite
        if hasattr(self,'env'): self.env.close()
        self.env = suite.load(self.domain,self.task,task_kwargs={'random': seed})

    def _obs(self, ts):
        if self.obs_mode=='pixels': return self.env.physics.render(self.screen_size,self.screen_size,camera_id=self.camera_id).transpose(2,0,1)
        return np.concatenate([np.asarray(ts.observation[k]).reshape(-1) for k in self.keys]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self._create(seed)
        return self._obs(self.env.reset()), {}

    def step(self, action):
        reward = 0.
        discount = 1.
        for i in range(self.action_repeat):
            ts = self.env.step(action)
            reward += discount * float(ts.reward or 0.)
            discount *= float(ts.discount)
            if ts.last(): break
        term = ts.last() and ts.discount == 0
        trunc = ts.last() and ts.discount != 0
        return self._obs(ts),reward,bool(term),bool(trunc),dict(discount=discount,base_steps=i+1)

    def close(self): self.env.close()

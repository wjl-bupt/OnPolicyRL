"""Explicit numeric encodings; categorical grids are never pixel-scaled."""
import gymnasium as gym
import numpy as np


class MiniGridEncoding(gym.ObservationWrapper):
    def __init__(self, env, mode='symbolic', mission_length=256):
        super().__init__(env)
        if mode not in ('symbolic', 'pixels'): raise ValueError('MiniGrid obs_mode: symbolic or pixels')
        self.mode, self.mission_length = mode, mission_length
        from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
        self.sizes = [max(d.values())+1 for d in (OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX)]
        shape = env.observation_space['image'].shape
        n = int(np.prod(shape)) if mode == 'pixels' else shape[0]*shape[1]*sum(self.sizes)
        # Fixed-width UTF-8 bytes + one-hot direction. No mission silently dropped.
        self.observation_space = gym.spaces.Box(0, 1, (n+4+mission_length,), dtype=np.float32)

    def observation(self, obs):
        image = obs['image']
        if self.mode == 'symbolic':
            image = np.concatenate([np.eye(n, dtype=np.float32)[image[..., i]].reshape(-1) for i,n in enumerate(self.sizes)])
        else: image = image.astype(np.float32).reshape(-1)/255.
        mission = obs['mission'].encode('utf-8')
        if len(mission) > self.mission_length: raise ValueError('mission exceeds configured encoding length')
        text = np.zeros(self.mission_length, np.float32)
        text[:len(mission)] = np.frombuffer(mission, np.uint8)/255.
        return np.concatenate([image, np.eye(4, dtype=np.float32)[obs['direction']], text])


class FloatObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        s = env.observation_space
        self.observation_space = gym.spaces.Box(s.low.astype(np.float32), s.high.astype(np.float32), dtype=np.float32)
    def observation(self, obs): return np.asarray(obs, dtype=np.float32)


class OneHotObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n = env.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (self.n,), dtype=np.float32)
    def observation(self, obs): return np.eye(self.n, dtype=np.float32)[obs]


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat):
        super().__init__(env)
        if type(repeat) is not int or repeat < 1: raise ValueError('action_repeat must be positive integer')
        self.repeat = repeat
    def step(self, action):
        total = 0.
        for n in range(self.repeat):
            obs, reward, term, trunc, info = self.env.step(action)
            total += reward
            if term or trunc: break
        return obs, total, term, trunc, dict(info, base_steps=n+1)

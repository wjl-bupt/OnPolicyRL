import numpy as np
import pytest
import torch
from gymnasium import spaces
from common.env.transforms import TensorTransforms, Moments
from protocol.masks import Masks


class Fake:
    num_envs=2
    device=torch.device('cpu')
    obs_space=spaces.Box(-100,100,(1,),dtype=np.float32)
    action_space=spaces.Box(-2,4,(1,),dtype=np.float32)
    def reset(self,seed=None): return torch.tensor([[1.],[3.]])
    def step(self,a):
        self.actions=a
        return torch.tensor([[10.],[4.]]),torch.tensor([2.,4.]),Masks(torch.tensor([False,False]),torch.tensor([True,False]),torch.tensor([True,True])),{'_transition_obs':torch.tensor([[5.],[4.]]),'_finished_episodes':[(2.,1)]}
    def close(self):pass


def test_running_moments_match_offline():
    m=Moments((2,),'cpu')
    x=torch.arange(18.).reshape(9,2)
    m.update(x[:3]);m.update(x[3:])
    torch.testing.assert_close(m.mean,x.double().mean(0))
    torch.testing.assert_close(m.var,x.double().var(0,unbiased=False))


def test_terminal_stack_and_action_rescale():
    e=TensorTransforms(Fake(),dict(frame_stack=3,action_mode='rescale'))
    assert e.reset().tolist()==[[1.,1.,1.],[3.,3.,3.]]
    obs,_,_,info=e.step(torch.tensor([[-1.],[1.]]))
    assert e.env.actions.tolist()==[[-2.],[4.]]
    assert obs.tolist()==[[10.,10.,10.],[3.,3.,4.]]
    assert info['_transition_obs'].tolist()==[[1.,1.,5.],[3.,3.,4.]]


def test_eval_frozen_raw_rewards_and_roundtrip(tmp_path):
    options=dict(normalize_observation=True,normalize_reward=True,reward_sign=True,observation_clip=10.)
    train=TensorTransforms(Fake(),options);train.reset();train.step(torch.zeros(2,1))
    path=tmp_path/'state.pt';torch.save(train.state_dict(),path)
    ev=TensorTransforms(Fake(),options,training=False)
    ev.load_state_dict(torch.load(path,weights_only=True))
    count=ev.obs_moments.count
    ev.reset();obs,r,m,info=ev.step(torch.zeros(2,1))
    assert ev.obs_moments.count==count
    assert r.tolist()==[2.,4.]
    assert info['_finished_episodes']==[(2.,1)]
    torch.testing.assert_close(ev.obs_moments.mean,train.obs_moments.mean)


def test_real_backends_share_minatar_contract():
    pytest.importorskip('gymnax')
    from common.env.factory import make_tensor_env
    for backend in ['gymnasium','jax']:
        env=make_tensor_env('MinAtar/Breakout-v0',backend=backend,num_envs=2,
            max_episode_steps=2,autoreset_mode='SameStep',wrappers={'frame_stack':2})
        try:
            obs=env.reset(seed=5)
            assert obs.dtype==torch.float32 and obs.shape==(2,8,10,10)
            assert obs.max()<=1
            env.step(torch.zeros(2,dtype=torch.int64))
            obs,r,m,info=env.step(torch.zeros(2,dtype=torch.int64))
            assert m.truncated.all() and m.valid.all()
            assert info['_transition_obs'].shape==obs.shape
        finally: env.close()


def test_minigrid_symbolic_keeps_mission_and_direction():
    pytest.importorskip('minigrid')
    import gymnasium as gym
    from common.env.observations import MiniGridEncoding
    env=MiniGridEncoding(gym.make('MiniGrid-Empty-5x5-v0'))
    try:
        raw,_=env.env.reset(seed=0)
        a=env.observation(raw)
        b=env.observation(dict(raw,mission='a different goal'))
        assert not np.array_equal(a,b)
        assert set(np.unique(a[:-256])) <= {0.,1.}
    finally:env.close()


def test_atari_repeat_stops_at_first_terminal(monkeypatch):
    import types,sys
    import jax
    import jax.numpy as jp
    from typing import NamedTuple
    class State(NamedTuple):
        env_state: object
    class Env:
        def __init__(self,*args,**kwargs):pass
        def image_space(self):return spaces.Box(0,255,(4,4,3),dtype=np.uint8)
        def action_space(self):return spaces.Discrete(2)
        def reset(self,key):return jp.zeros(1),State(jp.array(0))
        def step(self,s,a):return jp.zeros(1),State(s.env_state+1),jp.array(1.),jp.array(True),jp.array(False),{}
        def render(self,s):return jp.full((4,4,3),s,dtype=jp.uint8)
    monkeypatch.setitem(sys.modules,'jaxtari.core',types.SimpleNamespace(make=lambda name:Env()))
    monkeypatch.setitem(sys.modules,'jaxtari.wrappers',types.SimpleNamespace(AtariWrapper=Env))
    from common.env.jax_atari import atari_provider
    reset,step,*_=atari_provider('test',{'frame_skip':4,'screen_size':4,'noop_max':0})
    _,s=reset(jax.random.PRNGKey(0))
    obs,s,r,t,tr,_=jax.jit(step)(jax.random.PRNGKey(1),s,jp.array(0))
    assert float(r)==1 and int(s.env_state)==1 and bool(t)
    assert np.asarray(obs).min()==1

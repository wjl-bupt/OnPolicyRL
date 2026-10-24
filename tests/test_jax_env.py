import numpy as np
import pytest
import torch
pytest.importorskip('gymnax')
import jax
from common.env.make_env import make_env


@pytest.mark.parametrize('game,channels', [('Asterix',4),('Breakout',4),('Freeway',7),('Seaquest',10),('SpaceInvaders',6)])
def test_minatar_protocol_seed_and_truncation(game,channels):
    e=make_env(f'MinAtar/{game}-v0',backend='jax',num_envs=3,max_episode_steps=1)
    initial=e.reset(seed=7).clone()
    assert initial.shape==(3,channels,10,10) and e.action_space.n==6
    a=torch.zeros(3,dtype=torch.long)
    obs,reward,masks,info=e.step(a)
    assert masks.truncated.all() and masks.valid.all()
    assert not masks.terminated.any()
    assert len(info['_finished_episodes'])==3
    assert all(length==1 for _,length in info['_finished_episodes'])
    assert info['_transition_obs'].shape==initial.shape
    e.reset(seed=7)
    obs2,r2,m2,i2=e.step(a)
    torch.testing.assert_close(obs,obs2);torch.testing.assert_close(reward,r2)
    torch.testing.assert_close(info['_transition_obs'],i2['_transition_obs'])
    assert torch.isfinite(obs).all() and obs.max()<=1


def test_raw_successor_and_natural_termination_match_gymnax():
    import gymnax
    e=make_env('MinAtar/Breakout-v0',backend='jax',num_envs=2,max_episode_steps=500000,sticky_action_prob=0.0)
    e.reset(seed=3)
    g,p=gymnax.make('Breakout-MinAtar',use_minimal_action_set=False)
    p=p.replace(max_steps_in_episode=500000)
    found=False
    for _ in range(160):
        keys=jax.vmap(lambda k:jax.random.split(k,3)[1])(e._keys)
        raw=jax.jit(jax.vmap(g.step_env,in_axes=(0,0,0,None)))(keys,e._state.env_state,np.zeros(2,dtype=np.int32),p)
        obs,r,m,info=e.step(torch.zeros(2,dtype=torch.long))
        np.testing.assert_array_equal(info['_transition_obs'].permute(0,2,3,1).numpy(),np.asarray(raw[0]))
        np.testing.assert_array_equal(m.terminated.numpy(),np.asarray(raw[3]))
        if m.terminated.any():
            assert not m.truncated.any();found=True;break
    assert found


@pytest.mark.slow
def test_mjx_real_physics_and_time_limit():
    e=make_env('MJX/inverted_pendulum',backend='jax',num_envs=2,max_episode_steps=2)
    x=e.reset(seed=4).clone()
    assert x.shape==(2,4)
    y,r,m,i=e.step(torch.zeros(2,1))
    assert torch.isfinite(y).all() and not torch.equal(x,y)
    _,_,m,i=e.step(torch.zeros(2,1))
    assert m.truncated.all() and m.valid.all()
    assert len(i['_finished_episodes'])==2
    assert all(length==2 for _,length in i['_finished_episodes'])


def test_vector_eval_does_not_favor_short_episodes():
    from common.evaluate import evaluate
    from protocol.masks import Masks
    class Policy:
        def dist(self,x):return torch.distributions.Categorical(logits=torch.zeros(2,2))
    class Env:
        num_envs=2
        def reset(self,seed):self.t=0;return torch.zeros(2,1)
        def step(self,a):
            self.t+=1;ids=[0]+([1] if self.t%3==0 else [])
            info={'_finished_episode_indices':ids,'_finished_episodes':[(float(i*10),1 if i==0 else 3) for i in ids]}
            return torch.zeros(2,1),None,None,info
    result=evaluate(Policy(),Env,episodes=4)
    assert result['eval/ep_rew_mean']==5 and result['eval/n_episodes']==4

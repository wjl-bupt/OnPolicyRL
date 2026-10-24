import pytest
import torch
import numpy as np
from common.env.make_env import make_env
from workflow.experiment import Experiment, expand


@pytest.mark.parametrize('p',[0.0,1.0])
def test_jax_executed_action_and_reset(p):
    pytest.importorskip('gymnax')
    e=make_env('MinAtar/Freeway-v0',backend='jax',num_envs=8,max_episode_steps=2,sticky_action_prob=p)
    e.reset(seed=11)
    for _ in range(3):
        _,_,m,i=e.step(torch.full((8,),2))
        assert (i['_executed_actions']==(2 if p==0 else 0)).all()
    e.reset(seed=11)
    assert (e._state.last_action==0).all()


def test_default_probability_and_reproducibility():
    pytest.importorskip('gymnax')
    e=make_env('MinAtar/Freeway-v0',backend='jax',num_envs=512,max_episode_steps=2)
    assert e.provenance['sticky_action_prob']==.1
    e.reset(seed=23);_,_,_,a=e.step(torch.full((512,),2))
    actions=a['_executed_actions'].clone()
    assert 25<int((actions==0).sum())<80
    e.reset(seed=23);_,_,_,b=e.step(torch.full((512,),2))
    torch.testing.assert_close(actions,b['_executed_actions'])


@pytest.mark.parametrize('p',[0.0,.1,1.0])
def test_gymnasium_forwarding(p):
    pytest.importorskip('minatar')
    e=make_env('MinAtar/Breakout-v0',num_envs=1,sticky_action_prob=p)
    assert e.envs[0].unwrapped.game.sticky_action_prob==p
    e.close()


@pytest.mark.parametrize('p',[-.1,1.1,float('nan'),float('inf')])
def test_probability_rejected(p):
    with pytest.raises(ValueError):make_env('MinAtar/Breakout-v0',sticky_action_prob=p)


def test_defaults_are_only_injected_for_minatar(tmp_path):
    e=Experiment('ppo',['MinAtar/Breakout-v0','CartPole-v1'],1,False,tmp_path,{}, {},'mixed')
    jobs=expand(e)
    assert jobs[0].config['env_params']['sticky_action_prob']==.1
    assert 'env_params' not in jobs[1].config
    e.baseline={'env_params':{'sticky_action_prob':0.0}}
    assert expand(e)[0].config['env_params']['sticky_action_prob']==0.0

import copy
from dataclasses import asdict
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from architecture.actor_critic import PolicyConfig, build_policy
from common.checkpoint import save_checkpoint, load_checkpoint
from onpolicy._runtime import NativeLoop, make_buffer, estimates
from onpolicy.aae import AAE, AAEConfig, adaptive_lambdas, regression_accuracy, detached_distribution, joint_stat, build_policy as build_aae
from onpolicy.pae import PAE, PAEConfig
from protocol.masks import Masks

class Log:
    def __init__(self):self.values={};self.warnings=[]
    def record(self,k,v):self.values[k]=v
    def add(self,**kw):self.values.update(kw)
    def add_episode(self,*args):pass
    def warning(self,*args,**kw):self.warnings.append((args,kw))

class Stream:
    num_envs=2
    obs_space=spaces.Box(-100,100,(3,),dtype=np.float32)
    def __init__(self,continuous=False,boundaries=False):
        self.action_space=spaces.Box(-1,1,(2,),dtype=np.float32) if continuous else spaces.Discrete(3)
        self.boundaries=boundaries;self.t=0
    def observation(self):return th.tensor([[self.t,0,.1],[self.t,1,-.2]],dtype=th.float32)
    def reset(self,seed=None):self.t=0;return self.observation()
    def step(self,action):
        assert action.shape==(2,)+self.action_space.shape
        self.t+=1;obs=self.observation();final=obs.clone()
        term=th.tensor([False,self.boundaries and self.t%5==0])
        trunc=th.tensor([self.boundaries and self.t%4==0,False]);final[trunc,2]=3.
        valid=th.tensor([True,not(self.boundaries and self.t%5==1 and self.t>1)])
        return obs,th.tensor([.5,-.2]),Masks(term,trunc,valid),{'_transition_obs':final,'_finished_episodes':[]}

@pytest.mark.parametrize('variant',[1,2])
def test_aae_formulas(variant):
    entropy=th.tensor(-.7,dtype=th.float64)
    actor,critic,conv,acc=adaptive_lambdas(entropy,th.tensor(.8),th.tensor(.12),variant,.02,2.)
    c=1/(1+np.exp(-.7));a=.8*(1-2*(.12-.02))
    assert float(conv)==pytest.approx(c)
    assert float(acc)==pytest.approx(a)
    assert float(actor)==pytest.approx((1-.5*a)*c if variant==1 else 1-a*(1-.5*c))
    assert float(critic)==pytest.approx(1-.5*a*c)
    assert float(adaptive_lambdas(entropy,th.tensor(.8),th.tensor(10.),variant,.01,1.)[3])==0

def test_regression_metrics():
    y=th.tensor([1.,2.,4.,8.]);pred=y+2
    ev,_=regression_accuracy(y,pred,'ev',1e-8);r2,_=regression_accuracy(y,pred,'r2',1e-8)
    assert ev==1
    th.testing.assert_close(r2,1-4/y.var(unbiased=False))
    score,flag=regression_accuracy(th.ones(4),pred,'r2',1e-8)
    assert flag and score==0

@pytest.mark.parametrize('continuous',[False,True])
@pytest.mark.parametrize('variant',[1,2])
@pytest.mark.parametrize('metric',['r2','ev'])
def test_aae_updates_and_checkpoint(continuous,variant,metric,tmp_path):
    th.manual_seed(13)
    c=AAEConfig(num_envs=2,rollout_len=6,total_steps=24,ppo_epoch=2,num_mini_batch=2,aae_variant=variant,accuracy_metric=metric)
    env=Stream(continuous,True);net=PolicyConfig(net_arch=(8,));p=build_aae(env.obs_space,env.action_space,net,c)
    log=Log();algo=AAE(env,p,make_buffer(c,env),c,log=log)
    for update in range(2):
        algo._collect_rollout_();algo._compute_advantages_()
        actor=log.values['aae/lambda_actor'];critic=log.values['aae/lambda_critic']
        assert 0<=actor<=1 and .5<=critic<=1
        if update==0:assert critic==1
        adv,_=estimates(algo.rollout_buffer,c.gamma,actor)
        _,ret=estimates(algo.rollout_buffer,c.gamma,critic)
        th.testing.assert_close(algo.rollout_buffer.advantages,adv);th.testing.assert_close(algo.rollout_buffer.returns,ret)
        old=detached_distribution(p.dist(algo.flat('obs')[algo.indices()]));before=[v.clone() for v in p.parameters()]
        stats=algo._update_();algo.num_updates+=1
        new=p.dist(algo.flat('obs')[algo.indices()])
        expected=joint_stat(th.distributions.kl_divergence(old,new),new).mean()
        assert float(p.aae_kl)==pytest.approx(float(expected.detach()),abs=1e-7)
        assert all(np.isfinite(v) for v in stats.values())
        assert any(not th.equal(a,b) for a,b in zip(before,p.parameters()))
    path=tmp_path/'aae.pt';save_checkpoint(path,p,global_step=24,iteration=2)
    q=build_aae(env.obs_space,env.action_space,net,c);load_checkpoint(path,q)
    th.testing.assert_close(p.aae_regression,q.aae_regression);th.testing.assert_close(p.aae_kl,q.aae_kl)

@pytest.mark.parametrize('continuous',[False,True])
@pytest.mark.parametrize('boundaries',[False,True])
@pytest.mark.parametrize('use_vtrace',[False,True])
def test_pae_tail_accounting(continuous,boundaries,use_vtrace):
    th.manual_seed(21)
    c=PAEConfig(num_envs=2,rollout_len=6,partial_steps=2,total_steps=100,ppo_epoch=2,num_mini_batch=2,use_vtrace=use_vtrace)
    env=Stream(continuous,boundaries);p=build_policy(env.obs_space,env.action_space,PolicyConfig(net_arch=(8,)))
    algo=PAE(env,p,make_buffer(c,env),c,log=Log());consumed=set();newly_valid=0
    for iteration in range(5):
        previous=copy.deepcopy(algo._carry);before_step=algo.global_step;n=algo._collect_rollout_();b=algo.rollout_buffer
        start=0 if previous is None else len(previous['pending']);newly_valid+=int(b['valid'][start:].sum())
        assert algo.global_step-before_step==n==(6-start)*2
        if previous is not None:th.testing.assert_close(b['logprob'][:start],previous['fields']['logprob'])
        algo._compute_advantages_();valid=b['valid'].clone()
        for obs in b['obs'][algo._eligible]:
            key=tuple(obs[:2].tolist());assert key not in consumed;consumed.add(key)
        stats=algo._update_();algo.num_updates+=1
        th.testing.assert_close(b['valid'],valid)
        pending=0 if algo._carry is None else int(algo._carry['pending'].sum())
        assert newly_valid==len(consumed)+pending
        assert all(np.isfinite(v) for v in stats.values())
    if not boundaries:assert algo.global_step==28 and len(consumed)==20 and pending==8

@pytest.mark.parametrize('continuous',[False,True])
def test_pae_full_prefix_equals_ppo(continuous):
    c=PAEConfig(num_envs=2,rollout_len=6,partial_steps=6,total_steps=24,ppo_epoch=2,num_mini_batch=2)
    envs=[Stream(continuous,True),Stream(continuous,True)];th.manual_seed(17)
    p=build_policy(envs[0].obs_space,envs[0].action_space,PolicyConfig(net_arch=(8,)));q=copy.deepcopy(p)
    loops=[NativeLoop(envs[0],p,make_buffer(c,envs[0]),c,log=Log()),PAE(envs[1],q,make_buffer(c,envs[1]),c,log=Log())]
    for i in range(2):
        for algo in loops:
            th.manual_seed(101+i);algo._collect_rollout_();algo._compute_advantages_();algo._update_();algo.num_updates+=1
        th.testing.assert_close(loops[0].rollout_buffer.advantages,loops[1].rollout_buffer.advantages,atol=0,rtol=0)
        for a,b in zip(p.parameters(),q.parameters()):th.testing.assert_close(a,b,atol=0,rtol=0)
        assert loops[1]._carry is None

def test_pae_checkpoint(tmp_path):
    c=PAEConfig(num_envs=2,rollout_len=6,partial_steps=2,total_steps=100,ppo_epoch=1)
    env=Stream();net=PolicyConfig(net_arch=(8,));p=build_policy(env.obs_space,env.action_space,net)
    algo=PAE(env,p,make_buffer(c,env),c,log=Log());algo._collect_rollout_();algo._compute_advantages_();algo._update_()
    path=tmp_path/'pae.pt';save_checkpoint(path,p,global_step=12,iteration=1)
    env2=copy.deepcopy(env);q=build_policy(env2.obs_space,env2.action_space,net);state=load_checkpoint(path,q)
    other=PAE(env2,q,make_buffer(c,env2),c,log=Log());other.load_algorithm_state(state['algorithm_state'],trajectory_restored=True)
    for loop in (algo,other):th.manual_seed(99);loop._collect_rollout_();loop._compute_advantages_()
    th.testing.assert_close(algo.rollout_buffer.advantages,other.rollout_buffer.advantages)
    th.testing.assert_close(algo.flat('logprob'),other.flat('logprob'))
    other.load_algorithm_state(state['algorithm_state'],trajectory_restored=False)
    assert other._carry is None and other._last_obs is None
    assert other.log.values['pae/resume_discarded_pending']==8

def test_configs():
    assert asdict(PAEConfig(rollout_len=10))['partial_steps']==5
    for k in [0,11,1.2,True]:
        with pytest.raises(ValueError):PAEConfig(rollout_len=10,partial_steps=k)
    for kw in [dict(aae_variant=3),dict(accuracy_metric='mse'),dict(kl_scale=-1),dict(indicator_eps=0)]:
        with pytest.raises(ValueError):AAEConfig(**kw)


def test_vtrace_hand_calculation_no_double_actor_weight():
    from onpolicy.pae import vtrace_targets
    z=th.zeros(2,1);valid=th.ones(2,1,dtype=th.bool);false=~valid
    adv,target,rho,c=vtrace_targets(th.tensor([[1.],[2.]]),z,z,th.tensor([[.5],[2.]]).log(),z,false,false,valid,.9)
    th.testing.assert_close(target,th.tensor([[1.4],[2.]]))
    th.testing.assert_close(adv,th.tensor([[2.8],[2.]]))
    th.testing.assert_close(rho,th.tensor([[.5],[1.]]))
    # The first actor advantage must NOT be multiplied by rho=.5 again.
    assert not th.equal(adv,target)

@pytest.mark.parametrize('end',['terminal','timeout','invalid'])
def test_vtrace_boundaries(end):
    from onpolicy.pae import vtrace_targets
    z=th.zeros(2,1);valid=th.ones(2,1,dtype=th.bool);term=th.zeros_like(valid);trunc=term.clone()
    if end=='terminal':term[0]=True
    elif end=='timeout':trunc[0]=True
    else:valid[0]=False
    adv,target,_,_=vtrace_targets(th.tensor([[1.],[100.]]),z,th.tensor([[5.],[0.]]),z,z,term,trunc,valid,.9)
    expected={'terminal':1.,'timeout':5.5,'invalid':0.}[end]
    assert float(adv[0])==pytest.approx(expected)
    assert float(target[0])==pytest.approx(expected)


def test_vtrace_lambda_onpolicy_and_extreme_weights():
    from onpolicy.pae import vtrace_targets
    rewards=th.tensor([[1.],[2.],[3.]]);z=th.zeros_like(rewards);valid=th.ones_like(z,dtype=th.bool);false=~valid
    adv,target,_,_=vtrace_targets(rewards,z,z,z,z,false,false,valid,.9,trace_lambda=.5)
    th.testing.assert_close(target,th.tensor([[1+.45*(2+.45*3)],[2+.45*3],[3.]]))
    th.testing.assert_close(adv,th.tensor([[1+.9*(2+.45*3)],[2+.9*3],[3.]]))
    _,target,rho,_=vtrace_targets(rewards,z,z,z+10000,z,false,false,valid,.9)
    assert th.isfinite(target).all() and th.equal(rho,th.ones_like(z))
    assert PAEConfig().use_vtrace is False
    for kw in [dict(use_vtrace='false'),dict(vtrace_rho_clip=0),dict(vtrace_c_clip=2),dict(vtrace_lambda=1.1)]:
        with pytest.raises(ValueError):PAEConfig(**kw)

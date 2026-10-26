"""APO integration regressions: action axes, sampled extras, bootstraps, update."""
from dataclasses import asdict
import numpy as np
import pytest
import torch
from gymnasium import spaces
from architecture.actor_critic import PolicyConfig
from onpolicy.apo import APO, APOConfig, build_policy, make_apo_buffer
from protocol.masks import Masks

class Env:
    num_envs=2
    obs_space=spaces.Box(-100,100,(4,),dtype=np.float32)
    action_space=spaces.Box(-1,1,(3,),dtype=np.float32)
    def reset(self,seed=None):
        self.t=0
        return torch.zeros(2,4)
    def step(self,actions):
        assert actions.shape==(2,3)
        self.t+=1
        final=torch.full((2,4),float(self.t))
        obs=final.clone();obs[0]=-10
        return obs,torch.ones(2),Masks(torch.tensor([False,True]),torch.tensor([True,False]),torch.ones(2,dtype=torch.bool)),{'_transition_obs':final,'_finished_episodes':[]}

@pytest.mark.parametrize('k',[1,3])
def test_rollout_samples_probabilities_bootstrap_and_update(k):
    torch.manual_seed(8)
    cfg=APOConfig(ua_nums=k,num_envs=2,rollout_len=4,total_steps=16,num_mini_batch=2,ppo_epoch=2,epsilon=.13,clip_schedule='linear',clip_range_min=.03)
    env=Env();p=build_policy(env.obs_space,env.action_space,PolicyConfig(net_arch=(8,)),cfg)
    algo=APO(env,p,make_apo_buffer(cfg,env),cfg)
    algo._collect_rollout_();b=algo.rollout_buffer
    with torch.no_grad():
        torch.testing.assert_close(b['next_value'],p.predict_value(b['next_obs'].flatten(0,1)).reshape(4,2))
        a,lp,v=p.act(torch.zeros(2,4));assert a.shape==(2,3) and lp.shape==(2,)
    algo._compute_advantages_()
    # Every row ends an episode: timeout bootstraps its FINAL obs; termination does not.
    expected=b['reward']-b.values
    expected[:,0]+=cfg.gamma*b['next_value'][:,0]
    torch.testing.assert_close(b.advantages,expected)
    sample=next(b.sample());assert sample.uact.shape==(4,k,3)
    batch=algo._make_batch_(sample)
    torch.testing.assert_close(batch.logp,batch.old_log_prob)
    torch.testing.assert_close(batch.uact_logprobs,batch.uact_old_logprobs)
    for i in range(k):
        torch.testing.assert_close(batch.uact_logprobs[:,i],p.evaluate(sample.observations,sample.uact[:,i])[0])
    _,stats=algo.policy_loss(batch);assert stats['apo/uarr_loss']==0
    assert stats['train/clip_range']==pytest.approx(.08)
    before=[x.detach().clone() for x in p.parameters()]
    stats=algo._update_()
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(a,b) for a,b in zip(before,p.parameters()))
    algo._collect_rollout_();algo._compute_advantages_();algo._update_()


def test_config_and_unsupported_actions():
    assert asdict(APOConfig(ua_nums=4))['ua_nums']==4
    for value in [0,-1,True,1.5]:
        with pytest.raises(ValueError):APOConfig(ua_nums=value)
    with pytest.raises(ValueError,match='ua_nums'):
        build_policy(Env.obs_space,spaces.Discrete(2),PolicyConfig(),APOConfig())
    with pytest.raises(ValueError,match='flat Box or Discrete'):
        build_policy(Env.obs_space,spaces.MultiDiscrete([2,3]),PolicyConfig(),APOConfig())


@pytest.mark.parametrize('coef',[-1.,-2.,0.,0.3,2.])
def test_uarr_fixed_and_dynamic_coefficients(coef):
    from types import SimpleNamespace
    from onpolicy.apo import APOPolicyLoss
    old_prob=torch.tensor([[.2,.5],[.8,.4]])
    ratio=torch.tensor([[1.2,.7],[.9,1.4]])
    logp=(old_prob*ratio).log().requires_grad_()
    batch=SimpleNamespace(progress=0.,logp=torch.zeros(2),old_log_prob=torch.zeros(2),
                          advantages=torch.zeros(2),uact_logprobs=logp,
                          uact_old_logprobs=old_prob.log())
    loss,stats=APOPolicyLoss(uarr_coef=coef)(batch)
    weight=old_prob if coef<0 else torch.full_like(old_prob,coef)
    expected=(.5*weight*(ratio-1).square()).mean()
    torch.testing.assert_close(loss,expected)
    loss.backward()
    torch.testing.assert_close(logp.grad,weight*(ratio-1)*ratio/ratio.numel())
    assert stats['apo/uarr_coef']==pytest.approx(float(weight.mean()))
    assert stats['apo/uarr_loss']==pytest.approx(float(expected),abs=1e-7)


def test_uarr_config_reaches_loss():
    cfg=APOConfig(uarr_coef=.7,num_envs=2,rollout_len=4)
    env=Env();p=build_policy(env.obs_space,env.action_space,PolicyConfig(net_arch=(8,)),cfg)
    algo=APO(env,p,make_apo_buffer(cfg,env),cfg)
    assert asdict(cfg)['uarr_coef']==.7
    assert algo.policy_loss.uarr_coef==.7
    assert APOConfig().uarr_coef<0

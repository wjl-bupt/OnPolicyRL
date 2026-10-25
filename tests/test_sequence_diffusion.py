"""Contract, probability and synthetic rollout tests; no benchmark runs."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gymnasium import spaces
from architecture.actor_critic import PolicyConfig,build_policy
from architecture.backbones import resolve_normalization
from common.env.transforms import TensorTransforms
from protocol.masks import Masks


def config(**kwargs):
    return PolicyConfig(hidden_dim=8,num_blocks=1,attention_heads=2,context_length=3,
                        diffusion_steps=3,diffusion_hidden_dim=12,diffusion_blocks=1,diffusion_time_dim=8,**kwargs)


@pytest.mark.parametrize('gated',[False,True])
def test_transformer_causality_history_and_mode(gated):
    p=build_policy(spaces.Box(-1,1,(12,),dtype=np.float32),spaces.Discrete(3),
                   config(architecture='causal_transformer',transformer_gated=gated))
    x=torch.randn(2,12,requires_grad=True);changed=x.detach().clone();changed[:,-4:]+=10
    ex=p.pi_extractor
    torch.testing.assert_close(ex.sequence_features(x)[:,:-1],ex.sequence_features(changed)[:,:-1])
    p.dist(x).logits[:,0].sum().backward()
    assert x.grad[:,:4].abs().sum()>0  # actually reads earlier observations
    state={k:v.clone() for k,v in p.state_dict().items()}
    action,old,value=p.act(x);new,_,v=p.evaluate(x,action)
    torch.testing.assert_close(old,new);torch.testing.assert_close(value,v)
    p.eval();torch.testing.assert_close(new,p.evaluate(x,action)[0])
    for k,v in state.items():torch.testing.assert_close(v,p.state_dict()[k])


@pytest.mark.parametrize('algo',['ppo','ga2e','spo','rvl','dae','vmpo'])
def test_transformer_native_algorithm_heads(algo):
    from workflow.registry import get_algo
    entry=get_algo(algo);cfg=entry.config_cls()
    p=entry.build_policy(spaces.Box(-1,1,(12,),dtype=np.float32),spaces.Discrete(3),config(architecture='causal_transformer'),cfg)
    x=torch.randn(4,12);a,_,_=p.act(x);lp,ent,v=p.evaluate(x,a)
    (lp.mean()+v.square().mean()).backward()
    assert all(torch.isfinite(q.grad).all() for q in p.parameters() if q.grad is not None)


class Toy:
    """Two deterministic synthetic streams with independent episode boundaries."""
    num_envs=2;device=torch.device('cpu');next_step_autoreset=False
    obs_space=spaces.Box(-100,100,(4,),dtype=np.float32)
    action_space=spaces.Box(-2,2,(2,),dtype=np.float32)
    def reset(self,seed=None):self.t=0;return torch.zeros(2,4)
    def step(self,actions):
        self.t+=1;done=torch.tensor([self.t%2==0,self.t%3==0]);term=torch.zeros(2,dtype=torch.bool)
        final=torch.full((2,4),float(self.t));obs=final.clone();obs[done]=-self.t
        info={'_transition_obs':final,'_finished_episodes':[(float(self.t),self.t) for d in done if d],
              '_finished_episode_indices':done.nonzero().flatten().tolist()}
        return obs,1-actions.square().sum(-1),Masks(term,done,torch.ones(2,dtype=torch.bool)),info
    def close(self):pass


def test_history_terminal_and_reset_do_not_mix():
    env=TensorTransforms(Toy(),{'frame_stack':3,'action_mode':'rescale'});env.reset()
    env.step(torch.zeros(2,2));obs,_,_,info=env.step(torch.zeros(2,2))
    torch.testing.assert_close(obs[0],torch.full((12,),-2.))
    assert info['_transition_obs'][0].reshape(3,4)[:,0].tolist()==[0.,1.,2.]
    assert obs[1].reshape(3,4)[:,0].tolist()==[0.,1.,2.]


@pytest.mark.parametrize('architecture',['mlp','simba','causal_transformer'])
def test_diffusion_probabilities_bc_and_checkpoint(architecture):
    shape=(12,) if architecture=='causal_transformer' else (4,)
    cfg=config(architecture=architecture,action_head='diffusion')
    p=build_policy(spaces.Box(-1,1,shape,dtype=np.float32),spaces.Box(-1,1,(2,),dtype=np.float32),cfg)
    x=torch.randn(5,*shape);action,chain,old,value=p.sample_chain(x)
    new,v=p.evaluate_chain(x,chain)
    torch.testing.assert_close(new,old);torch.testing.assert_close(value,v)
    # Explicit multivariate diagonal-Gaussian log density at the last transition.
    features,_=p._features(x);d=p.transition(features,chain[:,-2],0)
    ref=-.5*(((chain[:,-1]-d.mean)/d.stddev).square()+2*d.stddev.log()+np.log(2*np.pi)).sum(-1)
    torch.testing.assert_close(new[:,-1],ref)
    assert p.posterior_std.min()>=p.min_std
    loss=-new.mean()+v.square().mean()+p.denoising_loss(x,torch.zeros(5,2))
    loss.backward();assert all(torch.isfinite(q.grad).all() for q in p.parameters() if q.grad is not None)
    p.optimizer.step()
    q=build_policy(spaces.Box(-1,1,shape,dtype=np.float32),spaces.Box(-1,1,(2,),dtype=np.float32),cfg);q.load_state_dict(p.state_dict())
    rng=torch.get_rng_state().clone();a=p.predict_action(x)
    torch.testing.assert_close(rng,torch.get_rng_state());torch.testing.assert_close(a,q.predict_action(x))
    with pytest.raises(RuntimeError):p.evaluate(x,action)
    with pytest.raises(RuntimeError):p.dist(x)


@pytest.mark.parametrize('architecture',['mlp','causal_transformer'])
def test_dppo_synthetic_rollout_update(architecture):
    from onpolicy.dppo import DPPO,DPPOConfig
    from onpolicy._runtime import make_buffer
    from protocol.buffer import Field
    cfg=DPPOConfig(num_envs=2,rollout_len=4,num_mini_batch=2,ppo_epoch=1,total_steps=8)
    network=config(architecture=architecture,action_head='diffusion')
    env=TensorTransforms(Toy(),resolve_normalization(vars(network),{})['wrappers'])
    p=build_policy(env.obs_space,env.action_space,network)
    buf=make_buffer(cfg,env,extra_schema={'denoising_chain':Field((4,2)),'denoising_logprob':Field((3,))})
    algo=DPPO(env,p,buf,cfg);algo._collect_rollout_()
    obs=algo.flat('obs');new,_=p.evaluate_chain(obs,algo.flat('denoising_chain'))
    torch.testing.assert_close(new,algo.flat('denoising_logprob'),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(algo.flat('action'),algo.flat('denoising_chain')[:,-1])
    algo._compute_advantages_();before=p.policy_head.output.weight.detach().clone()
    stats=algo._update_();assert np.isfinite(stats['loss/total'])
    assert not torch.equal(before,p.policy_head.output.weight)
    algo.log.close()


def test_diffusion_image_conditioning():
    for architecture in ['mlp','causal_transformer']:
        channels=12 if architecture=='causal_transformer' else 4
        cfg=config(architecture=architecture,action_head='diffusion',features_dim=8)
        p=build_policy(spaces.Box(0,1,(channels,10,10),dtype=np.float32),spaces.Box(-1,1,(2,),dtype=np.float32),cfg)
        x=torch.rand(2,channels,10,10);_,chain,lp,_=p.sample_chain(x)
        torch.testing.assert_close(lp,p.evaluate_chain(x,chain)[0])


def test_reject_unsupported_combinations(tmp_path):
    from workflow.experiment import expand,load_experiment
    import json
    data=dict(version=2,family='mujoco',algo='ppo',envs=['Pendulum-v1'],model={'action_head':'diffusion'})
    path=tmp_path/'cfg.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='require algo=dppo'):expand(load_experiment(path))
    data['algo']='dppo';data['model'].update(architecture='causal_transformer',context_length=3)
    path.write_text(json.dumps(data));job=expand(load_experiment(path))[0]
    assert job.config['env_params']['wrappers']['frame_stack']==3
    assert job.config['env_params']['wrappers']['action_mode']=='rescale'
    with pytest.raises(ValueError):resolve_normalization(data['model'],{'wrappers':{'frame_stack':4}})
    with pytest.raises(ValueError):build_policy(spaces.Box(-1,1,(4,),dtype=np.float32),spaces.Discrete(2),config(action_head='diffusion'))
    from onpolicy.dppo import DPPOConfig
    with pytest.raises(ValueError):DPPOConfig(ent_coef=.01)

def test_ga2e_update_with_explicit_history():
    from onpolicy.ga2e import GA2E,GA2EConfig
    from onpolicy._runtime import make_buffer
    from workflow.registry import get_algo
    class DiscreteToy(Toy):
        action_space=spaces.Discrete(3)
        def step(self,a):
            obs,_,m,info=super().step(torch.zeros(2,2))
            return obs,a.float(),m,info
    cfg=GA2EConfig(num_envs=2,rollout_len=4,ppo_epoch=1,num_mini_batch=2,total_steps=8,two_level=False)
    env=TensorTransforms(DiscreteToy(),{'frame_stack':3})
    p=get_algo('ga2e').build_policy(env.obs_space,env.action_space,config(architecture='causal_transformer'),cfg)
    algo=GA2E(env,p,make_buffer(cfg,env),cfg);algo._collect_rollout_();algo._compute_advantages_()
    assert np.isfinite(algo._update_()['loss/total']);algo.log.close()


def test_diffusion_evaluation_dispatch_and_no_rng_changes():
    from common.evaluate import evaluate
    p=build_policy(Toy.obs_space,spaces.Box(-1,1,(2,),dtype=np.float32),config(action_head='diffusion'))
    rng=torch.get_rng_state().clone()
    def factory():return TensorTransforms(Toy(),{'action_mode':'rescale'},training=False)
    a=evaluate(p,factory,episodes=4);b=evaluate(p,factory,episodes=4)
    assert a==b and a['eval/n_episodes']==4
    torch.testing.assert_close(rng,torch.get_rng_state())

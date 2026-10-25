"""Oracle tests: run with the original repo's SB3 2.7 environment.

Production oprl never imports SB3. This test reads the unmodified external source
and uses a thin policy protocol adapter to isolate algorithm implementation.
"""
import copy,hashlib,importlib.util,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gymnasium import spaces
from architecture.actor_critic import PolicyConfig
from onpolicy.ga2e import GA2E,GA2EConfig,build_policy
from onpolicy._runtime import make_buffer

SOURCE=Path(os.environ.get('GA2E_REFERENCE_SOURCE','/root/Posterior-GAE/src/algo/ga2e/ga2e.py'))
SOURCE_SHA='86e2b4950fa1701572467945ed0631de8c3c984bd7e0462c2332ce6375d02ec1'

class Log:
    def __init__(self):self.values={}
    def record(self,k,v,**kw):self.values[k]=v
    def add(self,**kw):self.values.update(kw)

class Adapter(torch.nn.Module):
    def __init__(self,p):super().__init__();self.inner=p;self.optimizer=p.optimizer
    def evaluate_actions(self,obs,act):
        lp,ent,v=self.inner.evaluate(obs,act);return v,lp,ent
    def predict_values(self,obs):return self.inner.predict_value(obs).unsqueeze(-1)
    def set_training_mode(self,mode):self.train(mode)

@pytest.fixture
def reference_class():
    sb3=pytest.importorskip('stable_baselines3')
    if not SOURCE.exists():pytest.skip('external reference not present')
    assert sb3.__version__=='2.7.0'
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'Reference changed: re-audit before updating hash'
    spec=importlib.util.spec_from_file_location('ga2e_external_oracle',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.GA2E


def pair(refclass,mode='rollout',continuous=False,**options):
    from stable_baselines3.common.buffers import RolloutBuffer
    torch.manual_seed(31);rng=np.random.default_rng(91);T,E=12,4
    obs_space=spaces.Box(-10,10,(5,),dtype=np.float32)
    action_space=spaces.Box(-1,1,(2,),dtype=np.float32) if continuous else spaces.Discrete(3)
    cfg=GA2EConfig(num_envs=E,rollout_len=T,total_steps=120,ppo_epoch=3,batch_size=7,
        lambda_refresh=mode,**options)
    network=PolicyConfig(net_arch=(8,8),lr=3e-4)
    policy=build_policy(obs_space,action_space,network,cfg)
    policy.lambda_ema.fill_(.61);policy.lambda_initialized.fill_(True)
    refpolicy=copy.deepcopy(policy);refpolicy.optimizer=torch.optim.Adam(refpolicy.parameters(),lr=network.lr,eps=network.adam_eps)
    env=SimpleNamespace(num_envs=E,obs_space=obs_space,action_space=action_space)
    buf=make_buffer(cfg,env);native=GA2E(env,policy,buf,cfg,log=Log());native.global_step=T*E
    obs=torch.tensor(rng.normal(size=(T,E,5)).astype(np.float32));last=torch.tensor(rng.normal(size=(E,5)).astype(np.float32))
    with torch.no_grad():
        actions,logp,values=policy.act(obs.reshape(-1,5))
    actions=actions.reshape((T,E)+action_space.shape);values=values.reshape(T,E);logp=logp.reshape(T,E)
    term=np.zeros((T,E),bool);trunc=np.zeros_like(term)
    term[3,0]=True;term[8,0]=True;term[5,1]=True;term[-1,2]=True
    trunc[4,2]=True;trunc[-1,3]=True
    successor=torch.cat((obs[1:],last[None]),0).clone()
    # Terminal observations are deliberately different from reset observations.
    successor[torch.tensor(term|trunc)]=torch.tensor(rng.normal(size=((term|trunc).sum(),5)).astype(np.float32))
    with torch.no_grad():next_values=policy.predict_value(successor.reshape(-1,5)).reshape(T,E)
    reward=rng.normal(size=(T,E)).astype(np.float32)
    arrays=dict(obs=obs,action=actions,logprob=logp,value=values,reward=torch.tensor(reward),
        next_obs=successor,next_value=next_values,terminated=torch.tensor(term),truncated=torch.tensor(trunc),valid=torch.ones(T,E,dtype=torch.bool))
    for name,a in arrays.items():buf[name][:T].copy_(a)
    buf.pos=T;native._last_obs=last
    rb=RolloutBuffer(T,obs_space,action_space,device='cpu',gamma=cfg.gamma,gae_lambda=cfg.gae_lambda,n_envs=E)
    rb.observations=obs.numpy().copy();rb.actions=actions.reshape(T,E,-1).numpy().copy()
    rb.values=values.numpy().copy();rb.log_probs=logp.numpy().copy()
    rb.rewards=reward+cfg.gamma*next_values.numpy()*(trunc & ~term)
    rb.episode_starts=np.zeros((T,E),np.float32);rb.episode_starts[0]=1;rb.episode_starts[1:]=(term|trunc)[:-1]
    rb.full=True;rb.pos=T
    ref=refclass.__new__(refclass)
    # Only protocol/config adaptation: all selection and update math runs from source.
    ref.policy=Adapter(refpolicy);ref.rollout_buffer=rb;ref._last_obs=last.numpy();ref._last_episode_starts=(term|trunc)[-1]
    for key in ['lambda_min','lambda_val','level2_points','val_every','ema_beta','two_level','lambda_refresh','var_penalty_beta','n_var_folds','grad_chunk_size','use_spo','kl_coef','value_ref_lambda','gamma','ent_coef','vf_coef','normalize_advantage','max_grad_norm','target_kl']:
        setattr(ref,key,getattr(cfg,key))
    ref._ema_lam=.61;ref._flat_gpu_cache=None;ref.num_timesteps=T*E;ref._current_progress_remaining=.6
    ref._logger=Log();ref.device=torch.device('cpu');ref.action_space=action_space
    ref.n_epochs=cfg.ppo_epoch;ref.batch_size=cfg.batch_size;ref._n_updates=0;ref.verbose=0
    ref.clip_range=lambda progress:native.policy_loss._current_eps(1-progress)
    ref.clip_range_vf=(lambda progress:cfg.vf_epsilon) if cfg.use_value_clipped else None
    ref._update_learning_rate=lambda opt:None
    return native,ref


@pytest.mark.parametrize('mode',['rollout','epoch'])
@pytest.mark.parametrize('continuous',[False,True])
@pytest.mark.parametrize('options',[
    {},{'grad_chunk_size':9},{'var_penalty_beta':.08,'n_var_folds':2},
    {'two_level':False,'lambda_min':.15},{'level2_points':7,'value_ref_lambda':.7},
    {'use_value_clipped':True,'vf_epsilon':.1}, {'kl_coef':.2},
    {'use_spo':True,'kl_coef':.2,'ent_coef':.02}, {'target_kl':1e-9},
    {'normalize_advantage':False}, {'val_every':1000},
])
def test_complete_updates_match_original(reference_class,mode,continuous,options):
    native,ref=pair(reference_class,mode,continuous,**options)
    # Fixed input delta/continuation and held-out ordering.
    delta,cont=ref._deltas_cont();nd,nc=native._deltas_cont()
    np.testing.assert_array_equal(delta,nd);np.testing.assert_array_equal(cont,nc)
    tr,va=ref._trajectory_split(cont);nt,nv=native.split()
    tr,va=np.flatnonzero(tr),np.flatnonzero(va)
    # Original applies this fallback in _select_lambda, not _trajectory_split.
    if min(len(tr),len(va))<8:tr=va=np.arange(48)
    np.testing.assert_array_equal(tr,nt);np.testing.assert_array_equal(va,nv)
    clip=.2 if mode=='epoch' else None
    sr,curve=ref._select_lambda(clip);sn,ncurve=native._select_lambda(clip)
    assert sr==sn and curve.keys()==ncurve.keys()
    np.testing.assert_allclose(list(curve.values()),list(ncurve.values()),atol=2e-7,rtol=2e-6)
    np.random.seed(719);ref.train();rng_after=np.random.get_state()
    np.random.seed(719);native._compute_advantages_();native._update_()
    np.testing.assert_array_equal(rng_after[1],np.random.get_state()[1]);assert rng_after[2]==np.random.get_state()[2]
    if mode=='rollout':
        # SB3 get() has flattened in environment-major order in place.
        ref_adv=ref.rollout_buffer.advantages.reshape(4,12).T
        ref_ret=ref.rollout_buffer.returns.reshape(4,12).T
    else:ref_adv=ref.rollout_buffer.advantages;ref_ret=ref.rollout_buffer.returns
    np.testing.assert_allclose(native.rollout_buffer.advantages.numpy(),ref_adv,atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(native.rollout_buffer.returns.numpy(),ref_ret,atol=2e-6,rtol=2e-6)
    assert float(native.policy.lambda_ema)==pytest.approx(ref._ema_lam,abs=1e-14)
    for new,old in zip(native.policy.parameters(),ref.policy.parameters()):
        torch.testing.assert_close(new,old,atol=0,rtol=0)
    assert len(native.policy.optimizer.state)==len(ref.policy.optimizer.state)
    for new_state,old_state in zip(native.policy.optimizer.state.values(),ref.policy.optimizer.state.values()):
        assert new_state.keys()==old_state.keys()
        for key in new_state:torch.testing.assert_close(new_state[key],old_state[key],atol=0,rtol=0)
    for key in ['ga2e/lambda_star','ga2e/lambda_used','ga2e/n_evals','ga2e/split_degenerate']:
        assert native.log.values[key]==pytest.approx(ref.logger.values[key],abs=1e-12)


def test_gradients_zero_ties_and_ema(reference_class):
    native,ref=pair(reference_class,grad_chunk_size=5)
    delta,cont=ref._deltas_cont();idx=np.arange(48)
    for clip in [None,.2]:
        for normalize in [False,True]:
            a=ref._gae(delta,cont,.93)
            torch.testing.assert_close(native.gradient(a,idx,normalize,clip),ref._policy_grad(a,idx,normalize,clip),atol=0,rtol=0)
    for lam in [.1,.95,1.,.2]:assert native._ema_update(lam)==ref._ema_update(lam)
    zeros=np.zeros_like(delta)
    native._deltas_cont=lambda:(zeros,cont);ref._deltas_cont=lambda:(zeros,cont)
    assert native._select_lambda()[0]==ref._select_lambda()[0]

"""Native algorithm equations, architecture, workflow, and checkpoint contracts."""
import ast
from pathlib import Path
import pytest
import torch as th
import gymnasium as gym
from architecture.actor_critic import PolicyConfig
from common.env.adapter import GymVectorAdapter
from common.logger import Logger
from common.checkpoint import save_checkpoint,load_checkpoint
from workflow.registry import get_algo

NAMES=['ppo','ga2e','spo','rvl','dae','vmpo']

def assemble(name,env_id='CartPole-v1'):
    env=GymVectorAdapter(gym.vector.SyncVectorEnv([lambda:gym.wrappers.RecordEpisodeStatistics(gym.make(env_id,max_episode_steps=12)) for _ in range(2)]))
    entry=get_algo(name)
    cfg=entry.config_cls(num_envs=2,rollout_len=24,total_steps=96,ppo_epoch=2,num_mini_batch=2,log_interval=1)
    policy=entry.build_policy(env.obs_space,env.action_space,PolicyConfig(net_arch=(16,)),cfg)
    return env,entry,cfg,policy

@pytest.mark.parametrize('name',NAMES)
def test_native_training_checkpoint_resume(name,tmp_path):
    th.set_num_threads(1);th.manual_seed(11)
    env,entry,cfg,policy=assemble(name)
    log=Logger(run_dir=tmp_path/'log')
    before={n:p.detach().clone() for n,p in policy.named_parameters()}
    try:
        summary=entry.train(cfg,env,policy,log)
        assert summary['global_step']==96
        assert all(th.isfinite(p).all() for p in policy.parameters())
        assert any(not th.equal(before[n],p) for n,p in policy.named_parameters())
        if name=='dae': assert not th.equal(before['advantage_head.weight'],policy.advantage_head.weight)
        if name=='rvl': assert not th.equal(before['relative_head.0.weight'],policy.relative_head[0].weight)
        if name=='ga2e': assert policy.lambda_initialized and cfg.lambda_min<=policy.lambda_ema<=1
        if name=='vmpo': assert (policy.duals>0).all() and not th.equal(before['duals'],policy.duals)
        save_checkpoint(tmp_path/'model.pt',policy,**{'global_step':96,'iteration':2})
        restored=entry.build_policy(env.obs_space,env.action_space,PolicyConfig(net_arch=(16,)),cfg)
        resume=load_checkpoint(tmp_path/'model.pt',restored)
        for k,v in policy.state_dict().items(): th.testing.assert_close(v,restored.state_dict()[k])
        cfg.total_steps=144
        assert entry.train(cfg,env,restored,log,resume=resume)['global_step']==144
    finally: env.close();log.close()

@pytest.mark.parametrize('name',['ppo','ga2e','spo','rvl','vmpo'])
def test_continuous_native(name,tmp_path):
    th.set_num_threads(1)
    env,entry,cfg,policy=assemble(name,'Pendulum-v1');log=Logger(run_dir=tmp_path)
    try:
        entry.train(cfg,env,policy,log)
        assert all(th.isfinite(p).all() for p in policy.parameters())
    finally:env.close();log.close()

def test_spo_formula():
    from onpolicy.spo import SoftPolicyLoss
    from types import SimpleNamespace
    a=th.tensor([-2.,1.,3.]);r=th.tensor([.7,1.,1.4],requires_grad=True)
    batch=SimpleNamespace(logp=r.log(),old_log_prob=th.zeros(3),advantages=a,progress=0.)
    loss,_=SoftPolicyLoss(.2)(batch)
    expected=-(a*r-a.abs()*(r-1).square()/.4).mean()
    th.testing.assert_close(loss,expected)
    th.testing.assert_close(th.autograd.grad(loss,r)[0],(-a+a.abs()*(r-1)/.2)/3)

def test_dae_discounted_equation_and_gradients():
    from onpolicy.dae import trajectory_residual
    r=th.tensor([1.,2.,3.]);a=th.tensor([.1,.2,.3],requires_grad=True);v=th.tensor([.4,.5,.6],requires_grad=True)
    got=trajectory_residual(r,a,v,th.tensor(2.),.9)
    want=th.stack([sum(.9**(j-i)*(r[j]-a[j]) for j in range(i,3))+.9**(3-i)*2-v[i] for i in range(3)])
    th.testing.assert_close(got,want)
    got.square().mean().backward();assert a.grad.abs().sum()>0 and v.grad.abs().sum()>0

def test_dae_centering_and_rvl_antisymmetry():
    for name in ['dae','rvl']:
        env,_,_,p=assemble(name)
        try:
            x=th.randn(5,4);y=th.randn(5,4)
            if name=='dae':
                prob=p.dist(x).probs.detach()
                th.testing.assert_close((p.advantages(x,prob)*prob).sum(-1),th.zeros(5),atol=1e-6,rtol=0)
            else:
                th.testing.assert_close(p.relative_value(x,y),-p.relative_value(y,x))
                th.testing.assert_close(p.relative_value(x,x),th.zeros(5))
        finally:env.close()

def test_rvl_terminal_targets():
    from onpolicy.rvl import pair_target
    ti=th.tensor([False,True,False,True]);tj=th.tensor([False,False,True,True])
    got=pair_target(th.ones(4)*3,th.ones(4),th.ones(4)*4,th.ones(4)*7,th.ones(4)*2,ti,tj,.9)
    th.testing.assert_close(got,th.tensor([5.6,5.6,4.7,0.]))

def test_vmpo_dual_direction_and_temperature():
    from onpolicy.vmpo import constraint_objective,temperature_objective
    for value in [.005,.02]:
        kl=th.tensor(value,requires_grad=True);alpha=th.tensor(1.,requires_grad=True)
        loss=constraint_objective(kl,alpha,.01);loss.backward()
        th.testing.assert_close(kl.grad,alpha.detach())
        th.testing.assert_close(alpha.grad,th.tensor(.01-value))
    a=th.tensor([1.,3.]);eta=th.tensor(.7,requires_grad=True)
    actual=temperature_objective(a,eta,.1)
    expected=.7*(.1+th.log(th.exp(a/.7).mean()))
    th.testing.assert_close(actual,expected);actual.backward();assert th.isfinite(eta.grad)

def test_no_external_rl_imports():
    root=Path(__file__).resolve().parents[1]/'src'
    banned=('stable_baselines3','sb3_contrib','oprl.vendor','algo.')
    for path in root.rglob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            imports=[node.module or ''] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            assert not any(name.startswith(banned) for name in imports),path
    for name in ('bgae','tgae','pgae'):assert not list(root.rglob(name))

@pytest.mark.parametrize('name',NAMES)
def test_existing_workflow_all_algorithms(name,tmp_path):
    from workflow.runner import run
    import yaml
    config={'algo':name,'env':'CartPole-v1','runs':1,'resume':True,'run_dir':str(tmp_path/'runs'),
        'baseline':{'algo_params':{'num_envs':2,'rollout_len':16,'total_steps':64,'ppo_epoch':1,'num_mini_batch':2},
        'network_params':{'net_arch':[16]},'monitor':{'eval_episodes':1,'ckpt_count':1,'eval_interval':100}},'space':{}}
    f=tmp_path/'exp.yaml';f.write_text(yaml.safe_dump(config))
    result=run(f);assert result[0]['status']=='done',result
    job=Path(result[0]['run_dir']);assert list((job/'model_pools').glob('*.pt'))
    assert (job/'metrics.jsonl').exists()
    assert run(f)[0]['status']=='skipped'
@pytest.mark.parametrize('name',['dae','rvl','vmpo','ga2e'])
def test_native_image_model_backward(name):
    from gymnasium.spaces import Box,Discrete
    import numpy as np
    entry=get_algo(name);cfg=entry.config_cls()
    p=entry.build_policy(Box(0,1,(3,10,10),dtype=np.float32),Discrete(3),PolicyConfig(features_dim=32),cfg)
    x=th.rand(4,3,10,10);actions,_,_=p.act(x)
    logp,entropy,value=p.evaluate(x,actions)
    loss=-logp.mean()+value.square().mean()
    if name=='dae':loss=loss+p.advantages(x,p.dist(x).probs.detach()).square().mean()
    if name=='rvl':loss=loss+p.relative_value(x,x.roll(1,0)).square().mean()
    loss.backward()
    assert all(th.isfinite(t.grad).all() for t in p.parameters() if t.grad is not None)
    assert any(t.grad is not None and t.grad.abs().sum()>0 for t in p.extractor.parameters())

def test_ga2e_gradient_and_segment_split():
    from onpolicy.ga2e import GA2E
    from onpolicy._runtime import make_buffer,estimates
    env,_,cfg,p=assemble('ga2e')
    try:
        loop=GA2E(env,p,make_buffer(cfg,env),cfg)
        loop._collect_rollout_()
        train,val=loop.split()
        # Force enough long trajectories to test no overlap without fallback.
        b=loop.rollout_buffer
        b['valid'].fill_(True);b['terminated'].zero_();b['truncated'].zero_()
        train,val=loop.split()
        assert not set(train.tolist()).intersection(val.tolist())
        adv=estimates(b,cfg.gamma,.6)[0]
        idx=loop.indices()
        params=list(p.parameters());before=[x.detach().clone() for x in params]
        gradient=loop.gradient(adv,idx)
        # At the rollout policy, clipped PPO gradient equals vanilla policy gradient.
        logp,_,_=p.evaluate(loop.flat('obs')[idx],loop.flat('action')[idx])
        expected=th.autograd.grad(-(logp*adv.flatten()[idx]).mean(),params,allow_unused=True)
        expected=th.cat([(th.zeros_like(v) if g is None else g).flatten() for v,g in zip(params,expected)])
        th.testing.assert_close(gradient,expected,atol=1e-5,rtol=1e-4)
        for a,v in zip(before,params):th.testing.assert_close(a,v)
        assert all(v.grad is None for v in params)
    finally:env.close()

def test_explicit_successor_bootstrap_masks():
    from onpolicy._runtime import make_buffer,estimates
    env,_,cfg,p=assemble('ppo')
    try:
        cfg.rollout_len=3
        b=make_buffer(cfg,env)
        b['reward'].fill_(1);b['value'].fill_(2);b['next_value'].fill_(10)
        b['valid'].fill_(True);b['terminated'].zero_();b['truncated'].zero_()
        b['truncated'][0]=True;b['terminated'][1]=True;b['valid'][2]=False
        adv,_=estimates(b,.9,.95)
        th.testing.assert_close(adv[0],th.tensor([8.,8.]))
        th.testing.assert_close(adv[1],th.tensor([-1.,-1.]))
        th.testing.assert_close(adv[2],th.zeros(2))
    finally:env.close()

def test_same_step_uses_final_not_reset_observation():
    import numpy as np
    from gymnasium.vector import AutoresetMode
    class Tiny(gym.Env):
        observation_space=gym.spaces.Box(-100,100,(1,),dtype=np.float32)
        action_space=gym.spaces.Discrete(2)
        def reset(self,seed=None,options=None):super().reset(seed=seed);return np.array([0.],dtype=np.float32),{}
        def step(self,action):return np.array([7.],dtype=np.float32),1.,False,True,{}
    env=GymVectorAdapter(gym.vector.SyncVectorEnv([Tiny],autoreset_mode=AutoresetMode.SAME_STEP))
    try:
        env.reset(seed=0)
        obs,_,m,info=env.step(th.tensor([0]))
        assert obs.item()==0 and info['_transition_obs'].item()==7 and m.valid.item()
    finally:env.close()

def test_episode_stats_do_not_count_unfinished_vector_slots():
    import numpy as np
    info={'episode':{'r':np.array([12.,0.]),'l':np.array([5,0])},'_episode':np.array([True,False])}
    assert GymVectorAdapter._finished_episodes(info,None,th.tensor([True,True]))==[(12.,5)]

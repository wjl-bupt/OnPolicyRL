import copy,json
import numpy as np
import pytest
import torch
from gymnasium import spaces
from architecture.actor_critic import PolicyConfig,build_policy
from architecture.backbones import SimBaBlock,BroBlock,SpectralLinear,resolve_normalization

@pytest.mark.parametrize('name',['mlp','simba','bronet','spectralnet'])
@pytest.mark.parametrize('continuous',[False,True])
def test_policy_contract_and_restore(name,continuous):
    cfg=PolicyConfig(architecture=name,hidden_dim=12,num_blocks=1,actor_hidden_dim=12,critic_hidden_dim=16)
    obs=spaces.Box(-1,1,(5,),dtype=np.float32)
    actions=spaces.Box(-1,1,(2,),dtype=np.float32) if continuous else spaces.Discrete(3)
    p=build_policy(obs,actions,cfg);x=torch.randn(7,5)
    act,lp,v=p.act(x);logp,ent,value=p.evaluate(x,act)
    torch.testing.assert_close(lp,logp);torch.testing.assert_close(v,value)
    loss=-(logp+ent*.01).mean()+value.square().mean();loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in p.parameters() if t.grad is not None)
    p.optimizer.step();q=build_policy(obs,actions,cfg);q.load_state_dict(p.state_dict())
    torch.testing.assert_close(p.predict_value(x),q.predict_value(x))

@pytest.mark.parametrize('name',['simba','bronet','spectralnet'])
def test_ga2e_per_sample_grad_is_pure(name):
    p=build_policy(spaces.Box(-1,1,(4,),dtype=np.float32),spaces.Discrete(2),PolicyConfig(architecture=name,hidden_dim=8,num_blocks=1))
    class Logp(torch.nn.Module):
        def __init__(self,p):super().__init__();self.policy=p
        def forward(self,x):return self.policy.dist(x).log_prob(torch.zeros(x.shape[:-1],dtype=torch.long))
    model=Logp(p);params=dict(model.named_parameters());before=copy.deepcopy(model.state_dict())
    def fn(params,x):return torch.func.functional_call(model,params,(x.unsqueeze(0),)).sum()
    grads=torch.func.vmap(torch.func.grad(fn),in_dims=(None,0))(params,torch.randn(3,4))
    assert all(torch.isfinite(g).all() for g in grads.values())
    for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k])

def test_residual_paths_and_sn():
    for block in [SimBaBlock(6),BroBlock(6)]:
        for p in block.parameters():p.data.zero_()
        x=torch.randn(4,6);torch.testing.assert_close(block(x),x)
    layer=SpectralLinear(6,6);w=layer.weight/torch.linalg.matrix_norm(layer.weight,ord=2)
    torch.testing.assert_close(torch.linalg.svdvals(w).max(),torch.tensor(1.))

def test_normalization_resolution():
    assert resolve_normalization({'architecture':'simba'}, {})['wrappers']['normalize_observation']
    assert not resolve_normalization({'architecture':'simba','rsnorm':False}, {})['wrappers']['normalize_observation']
    with pytest.raises(ValueError):resolve_normalization({'architecture':'simba'}, {'wrappers':{'normalize_observation':False}})

@pytest.mark.parametrize('name',['simba','bronet','spectralnet'])
def test_image_and_auxiliary_heads(name):
    from architecture.auxiliary import build_special
    from types import SimpleNamespace
    cfg=PolicyConfig(architecture=name,hidden_dim=8,num_blocks=1,actor_hidden_dim=8,critic_hidden_dim=12,features_dim=16)
    obs=spaces.Box(0,1,(4,10,10),dtype=np.float32);act=spaces.Discrete(4);x=torch.rand(3,4,10,10)
    for kind in ['ga2e','dae','rvl','vmpo']:
        p=build_special(obs,act,cfg,SimpleNamespace(value_head='linear',dual_lr=1e-3),kind)
        assert p.act(x)[0].shape==(3,)
        if kind=='dae':assert p.advantages(x,torch.full((3,4),.25)).shape==(3,4)
        if kind=='rvl':torch.testing.assert_close(p.relative_value(x,x),torch.zeros(3))

def test_local_override_snapshot(tmp_path):
    from workflow.overrides import materialize
    source=tmp_path/'family.yaml';source.write_text('version: 2\nfamily: minatar\nalgo: ppo\nenvs: [MinAtar/Asterix-v0]\n')
    before=source.read_text()
    result=materialize(source,['algo=[ppo, ga2e]','algorithms.ppo.train.gae_lambda=0.95','model.architecture=simba'],tmp_path/'snapshots')
    data=json.loads(open(result).read());assert data['algo']==['ppo','ga2e']
    from workflow.experiment import load_experiment,expand
    jobs=expand(load_experiment(result));assert len(jobs)==2
    assert all(j.config['env_params']['wrappers']['normalize_observation'] for j in jobs)
    assert source.read_text()==before
    assert result==materialize(source,['algo=[ppo, ga2e]','algorithms.ppo.train.gae_lambda=0.95','model.architecture=simba'],tmp_path/'snapshots')

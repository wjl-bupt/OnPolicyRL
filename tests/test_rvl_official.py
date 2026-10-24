"""Numerical oracles from the author's pinned official RVL repository."""
import ast
from pathlib import Path
import torch
from onpolicy.rvl import pair_target,RVL
from onpolicy._runtime import make_buffer
from test_native_algorithms import assemble


def official_functions():
    ns={'torch':torch}
    for name in ['value_init.py','target_1.py','gae.py']:
        text=(Path(__file__).parent/'reference'/name).read_text()
        tree=ast.parse(text)
        tree.body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
        exec(compile(tree,name,'exec'),ns)
    return ns


def test_official_one_step_all_terminal_combinations():
    torch.manual_seed(0);size=64
    class Agent:
        def encoding_to_rv(self,a,b):return torch.tanh(a-b).sum(-1)
    agent=Agent()
    i={'obs':torch.randn(size,4),'encoded_obs':torch.randn(size,4),'encoded_next_obs':torch.randn(size,4),'rewards':torch.randn(size),'next_done':torch.arange(size)%2==0}
    j={'obs':torch.randn(size,4),'encoded_obs':torch.randn(size,4),'encoded_next_obs':torch.randn(size,4),'rewards':torch.randn(size),'next_done':torch.arange(size)%4<2}
    reference,_,_=official_functions()['rv_1_step_target'](i,j,agent,.99)
    ours=pair_target(i['rewards'],j['rewards'],agent.encoding_to_rv(i['encoded_next_obs'],j['encoded_next_obs']),agent.encoding_to_rv(i['encoded_obs'],j['encoded_next_obs']),agent.encoding_to_rv(i['encoded_next_obs'],j['encoded_obs']),i['next_done'],j['next_done'],.99)
    torch.testing.assert_close(ours,reference)


def test_official_rgae_at_rollout_boundary():
    env,_,cfg,p=assemble('rvl')
    try:
        cfg.rollout_len=8;cfg.optimal_offset=False
        b=make_buffer(cfg,env);b['valid'].fill_(True)
        states=torch.randn(9,2,4);b['obs'].copy_(states[:-1]);b['next_obs'].copy_(states[1:]);b['reward'].copy_(torch.randn(8,2))
        class Agent:
            def get_rv(self,a,b):return p.relative_value(a,b)
        data={'obs':b.obs,'next_obs':b['next_obs'],'rewards':b['reward'],'done':torch.zeros(8,2),'next_done':torch.zeros(8,2)}
        official_functions()['gae_fast'](Agent(),data,cfg.gae_lambda,cfg.gamma,'zeros',None)
        algo=RVL(env,p,b,cfg);algo._compute_advantages_()
        torch.testing.assert_close(b.advantages,data['advantages'])
        assert not any('value_head' in n for n,_ in p.named_parameters())
    finally:env.close()


def test_ga2e_chunked_gradient_matches_full_fold():
    from onpolicy.ga2e import GA2E
    from onpolicy._runtime import estimates
    env,_,cfg,p=assemble('ga2e')
    try:
        algo=GA2E(env,p,make_buffer(cfg,env),cfg);algo._collect_rollout_()
        adv=estimates(algo.rollout_buffer,.99,.7)[0];idx=algo.indices()
        cfg.grad_chunk_size=0;whole=algo.gradient(adv,idx,True)
        cfg.grad_chunk_size=7;chunked=algo.gradient(adv,idx,True)
        torch.testing.assert_close(whole,chunked,atol=1e-6,rtol=1e-4)
    finally:env.close()

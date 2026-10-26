import copy
import numpy as np
import pytest
import torch
from gymnasium import spaces
from architecture.actor_critic import PolicyConfig
from onpolicy.apo import APO,APOConfig,build_policy,make_apo_buffer
from onpolicy.ga2e import GA2E,GA2EConfig,build_policy as build_ga2e
from onpolicy._runtime import make_buffer
from onpolicy._review_metrics import sample_independent,joint_kl
from protocol.masks import Masks


class Log:
    def __init__(self):self.values={}
    def record(self,k,v):self.values[k]=v
    def add(self,**kw):self.values.update(kw)
    def dump(self,*args):pass


class Env:
    num_envs=4
    obs_space=spaces.Box(-10,10,(4,),dtype=np.float32)
    def __init__(self,discrete=True):self.action_space=spaces.Discrete(4) if discrete else spaces.Box(-1,1,(2,),dtype=np.float32)
    def reset(self,seed=None):self.t=0;return torch.zeros(4,4)
    def step(self,action):
        assert action.shape==((4,) if hasattr(self.action_space,'n') else (4,2))
        self.t+=1;obs=torch.full((4,4),float(self.t%5))
        done=torch.full((4,),self.t%4==0,dtype=torch.bool)
        return obs,torch.arange(4).float()+1,Masks(done,torch.zeros_like(done),torch.ones_like(done)),{'_finished_episodes':[],'_transition_obs':obs}


def make(kind='ga2e',discrete=True,**options):
    env=Env(discrete);cls=GA2EConfig if kind=='ga2e' else APOConfig
    cfg=cls(num_envs=4,rollout_len=12,total_steps=96,ppo_epoch=2,num_mini_batch=3,**options)
    policy=(build_ga2e if kind=='ga2e' else build_policy)(env.obs_space,env.action_space,PolicyConfig(net_arch=(8,)),cfg)
    algo=(GA2E if kind=='ga2e' else APO)(env,policy,(make_buffer if kind=='ga2e' else make_apo_buffer)(cfg,env),cfg,log=Log())
    return algo


@pytest.mark.parametrize('k',[1,2,3])
def test_discrete_anchors_unique_exclude_executed_and_logprob(k):
    a=make('apo',ua_nums=k);a._collect_rollout_();b=a.rollout_buffer
    assert b['uact'].dtype==torch.int64
    assert not (b['uact']==b['action'].unsqueeze(-1)).any()
    for row in b['uact'].reshape(-1,k):assert len(row.unique())==k
    a._compute_advantages_()
    batch=a._make_batch_(next(b.sample()))
    torch.testing.assert_close(batch.logp,batch.old_log_prob)
    torch.testing.assert_close(batch.uact_logprobs,batch.uact_old_logprobs)
    a._update_()
    assert 'review/independent/ratio/p95' in a.log.values


@pytest.mark.parametrize('n,k',[(1,1),(2,2),(4,4)])
def test_discrete_bounds(n,k):
    with pytest.raises(ValueError,match='ua_nums'):
        build_policy(Env.obs_space,spaces.Discrete(n),PolicyConfig(),APOConfig(ua_nums=k))


@pytest.mark.parametrize('kind',['apo','ga2e'])
@pytest.mark.parametrize('discrete',[True,False])
def test_metrics_do_not_change_rng_or_parameters(kind,discrete):
    torch.manual_seed(19);a=make(kind,discrete,review_metrics=False)
    a._collect_rollout_();b=copy.deepcopy(a);b.config.review_metrics=True
    rng=torch.get_rng_state();nrng=np.random.get_state()
    a._compute_advantages_();a._update_();end=torch.get_rng_state();nend=np.random.get_state()
    torch.set_rng_state(rng);np.random.set_state(nrng)
    b._compute_advantages_();b._update_()
    assert torch.equal(end,torch.get_rng_state())
    assert np.array_equal(nend[1],np.random.get_state()[1])
    for p,q in zip(a.policy.parameters(),b.policy.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)


@pytest.mark.parametrize('mode',['rollout','epoch'])
@pytest.mark.parametrize('degenerate',[False,True])
def test_candidate_only_never_updates_reference(mode,degenerate):
    a=make(update_data='candidate',lambda_refresh=mode,val_every=1000 if degenerate else 5)
    a._collect_rollout_();a._compute_advantages_()
    seen=[];original=a.policy.evaluate
    # Inspect update indices directly; search may intentionally use fallback data.
    a._update_()
    assert len(a._update_indices)>0
    assert set(a._update_indices)==set(a._candidate_indices)
    assert not set(a._update_indices)&set(a._reference_indices)
    assert set(a._critic_update_indices)==set(a.indices().cpu().tolist())
    assert a.log.values['review/update_reference_overlap']==0


def test_optional_reference_diagnostics_no_training_change():
    torch.manual_seed(23);a=make(review_metrics=True);a._collect_rollout_();b=copy.deepcopy(a)
    b.config.review_reference_interval=1;b.config.review_used_lambda=True
    state=np.random.get_state();rng=torch.get_rng_state()
    a._compute_advantages_();a._update_()
    np.random.set_state(state);torch.set_rng_state(rng)
    b._compute_advantages_();b._update_()
    for p,q in zip(a.policy.parameters(),b.policy.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert 'review_reference/split_half/cosine' in b.log.values


def test_independent_sampling_does_not_consume_global_rng():
    for d in [torch.distributions.Normal(torch.zeros(5,2),torch.ones(5,2)),torch.distributions.Categorical(logits=torch.zeros(5,4))]:
        state=torch.get_rng_state();sample_independent(d,8,42)
        assert torch.equal(state,torch.get_rng_state())
        assert torch.equal(joint_kl(d,d),torch.zeros(5))






@pytest.mark.parametrize('mode',['rollout','epoch'])
@pytest.mark.parametrize('target_kl',[None,1e-15])
def test_candidate_critic_consumes_all_targets(monkeypatch,mode,target_kl):
    import onpolicy.ga2e as module
    a=make(update_data='candidate',lambda_refresh=mode,target_kl=target_kl)
    a._collect_rollout_();a._compute_advantages_()
    targets=[];original=module.F.mse_loss
    def capture(target,pred,*args,**kw):
        targets.append(target.detach().clone());return original(target,pred,*args,**kw)
    monkeypatch.setattr(module.F,'mse_loss',capture)
    a._update_()
    expected=a.flat('returns')[a.indices()].repeat(a.config.ppo_epoch).sort().values
    torch.testing.assert_close(torch.cat(targets).sort().values,expected)


def test_candidate_actor_excludes_reference_advantages():
    a=make(update_data='candidate',vf_coef=0.,review_metrics=False)
    a._collect_rollout_();a._compute_advantages_();b=copy.deepcopy(a)
    b.rollout_buffer.advantages.reshape(-1)[torch.tensor(b._reference_indices)]+=1000.
    rng=np.random.get_state();a._update_();np.random.set_state(rng);b._update_()
    for p,q in zip(a.policy.parameters(),b.policy.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_optional_apo_gradient_diagnostic_preserves_update():
    a=make('apo',review_gradient_interval=0);a._collect_rollout_();a._compute_advantages_();b=copy.deepcopy(a)
    b.config.review_gradient_interval=1
    a._update_();b._update_()
    for p,q in zip(a.policy.parameters(),b.policy.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert 'review/regularizer_pg_norm_ratio' in b.log.values


def test_probe_is_one_use_and_does_not_modify_parameters():
    a=make();a._collect_rollout_();a._compute_advantages_();b=copy.deepcopy(a)
    obs=torch.zeros(3,4);actions=torch.zeros(3,dtype=torch.long)
    b.set_review_probe(obs,actions,torch.ones(3),source='independent-test-data',seed=9)
    rng=np.random.get_state();a._update_();np.random.set_state(rng);b._update_()
    for p,q in zip(a.policy.parameters(),b.policy.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert b._review_probe is None
    assert 'review_probe/clipped_surrogate_gain' in b.log.values


def test_review_counters_sum_and_candidate_tables_keep_last():
    from common.logger import _agg_for
    assert _agg_for('review_search/search_calls')=='sum'
    assert _agg_for('review_search/gradient_evaluations')=='sum'
    assert _agg_for('review/optimizer_steps')=='sum'
    assert _agg_for('review_candidates/lambda_0.950_score')=='last'

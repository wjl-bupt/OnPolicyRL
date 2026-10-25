"""Self-owned auxiliary network heads, conforming to the native Policy contract."""
import torch as th
from torch import nn
from architecture.actor_critic import ActorCritic, build_policy
from architecture.init import layer_init

class AdvantagePolicy(ActorCritic):
    def advantages(self, obs, probabilities):
        _, vf = self._features(obs)
        raw = self.advantage_head(vf)
        return raw - (probabilities.detach()*raw).sum(-1,keepdim=True)

class RelativeValuePolicy(ActorCritic):
    def _value(self, feat_v):
        # RVL has no absolute-value regressor; R-GAE supplies relative values.
        return th.zeros(feat_v.shape[0], device=feat_v.device)
    def relative_value(self, first, second):
        _, a=self._features(first)
        _, b=self._features(second)
        # Odd, bias-free head gives Delta(i,j) = -Delta(j,i), Delta(i,i)=0.
        return self.relative_head(a-b).squeeze(-1)

def build_special(obs_space, act_space, network, algo, kind):
    if network.action_head == 'diffusion':raise ValueError('Diffusion requires dppo, not an auxiliary Gaussian/categorical policy')
    if kind=='dae' and type(act_space).__name__!='Discrete':
        raise ValueError('DAE requires discrete actions')
    base=build_policy(obs_space,act_space,network)
    if kind in ('dae','rvl'):
        klass=AdvantagePolicy if kind=='dae' else RelativeValuePolicy
        extractors=base.extractor if base.share else (base.pi_extractor,base.vf_extractor)
        policy=klass(extractors,base.policy_head,base.value_head,base.share)
        policy.pi_post, policy.vf_post = base.pi_post, base.vf_post
        width=base.value_head.in_features
        if kind=='dae':
            policy.advantage_head=layer_init(nn.Linear(width,act_space.n),gain=1.0)
        else:
            if algo.value_head == 'linear':
                policy.relative_head=nn.Sequential(layer_init(nn.Linear(width,1,bias=False),gain=1.0))
            elif algo.value_head == 'mlp_symmetric':
                policy.relative_head=nn.Sequential(nn.Linear(width,96,bias=False),nn.Tanh(),nn.Linear(96,64,bias=False),nn.Tanh(),nn.Linear(64,64,bias=False),nn.Tanh(),nn.Linear(64,1,bias=False))
            else: raise ValueError('Unknown RVL value_head')
            policy.value_head=nn.Identity()
    else: policy=base
    if kind=='ga2e':
        policy.register_buffer('lambda_ema',th.tensor(0.0,dtype=th.float64))
        policy.register_buffer('lambda_initialized',th.tensor(False))
    if kind=='vmpo':
        policy.duals=nn.Parameter(th.ones(3))
        ordinary=[p for p in policy.parameters() if p is not policy.duals]
        policy.optimizer=network.optimizer_cls([{'params':ordinary},{'params':[policy.duals],'lr':algo.dual_lr}],lr=network.lr,eps=network.adam_eps)
    else:
        policy.optimizer=network.optimizer_cls(policy.parameters(),lr=network.lr,eps=network.adam_eps)
    return policy

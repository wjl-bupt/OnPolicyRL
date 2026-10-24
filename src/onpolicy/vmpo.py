"""Native V-MPO: weighted likelihood, temperature and exact distribution KL duals."""
from dataclasses import dataclass
import copy,math
import torch as th
from torch.distributions import Categorical,Normal,kl_divergence
from onpolicy.ppo import PPOConfig
from onpolicy._runtime import NativeLoop,run_algorithm

@dataclass
class VMPOConfig(PPOConfig):
    epsilon_eta: float = .1
    epsilon_kl: float = .01
    epsilon_mu: float = .01
    epsilon_sigma: float = .00001
    dual_lr: float = .001
    def __post_init__(self):
        super().__post_init__()
        if min(self.epsilon_eta,self.epsilon_kl,self.epsilon_mu,self.epsilon_sigma,self.dual_lr)<=0: raise ValueError('V-MPO constraints must be positive')

def temperature_objective(advantages,eta,bound):
    return eta*(bound+th.logsumexp(advantages.detach()/eta,0)-math.log(len(advantages)))

def constraint_objective(kl,alpha,bound):
    return alpha.detach()*kl+alpha*(bound-kl.detach())

class VMPO(NativeLoop):
    def _update_(self):
        c=self.config;p=self.policy
        old=copy.deepcopy(p).eval()
        for par in old.parameters(): par.requires_grad_(False)
        obs=self.flat('obs'); actions=self.flat('action')
        for idx in self.batches():
            logp,entropy,value=p.evaluate(obs[idx],actions[idx])
            advantages=self.flat('advantages')[idx]
            selected=advantages.topk(max(1,len(idx)//2)).indices
            top=advantages[selected]
            eta=p.duals[0]
            weights=th.softmax(top/eta.detach(),0).detach()
            pg=-(weights*logp[selected]).sum()
            temperature=temperature_objective(top,eta,c.epsilon_eta)
            with th.no_grad(): prior=old.dist(obs[idx])
            current=p.dist(obs[idx])
            if isinstance(current,Categorical):
                kls=[kl_divergence(prior,current).mean()]; bounds=[c.epsilon_kl]
            elif isinstance(current,Normal):
                kls=[kl_divergence(prior,Normal(current.loc,prior.scale)).sum(-1).mean(),
                     kl_divergence(prior,Normal(prior.loc,current.scale)).sum(-1).mean()]
                bounds=[c.epsilon_mu,c.epsilon_sigma]
            else: raise TypeError('V-MPO requires categorical or diagonal Gaussian native heads')
            constraint=sum(constraint_objective(kl,p.duals[i+1],bound) for i,(kl,bound) in enumerate(zip(kls,bounds)))
            vf=.5*(value-self.flat('returns')[idx]).square().mean()
            loss=pg+temperature+constraint+c.vf_coef*vf-c.ent_coef*entropy.mean()
            self.step_loss(loss)
            with th.no_grad(): p.duals.clamp_(min=1e-6)
            stats={'loss/policy':float(pg.detach()),'loss/value':float(vf.detach()),'vmpo/eta':float(p.duals[0].detach()),'train/kl':sum(float(k.detach()) for k in kls)}
            self.log.add(**stats)
        return stats

def train(cfg,env,policy,log=None,monitor=None,resume=None):
    return run_algorithm(VMPO,cfg,env,policy,log,monitor,resume)


# Algorithm discovery contract (no central name/class mapping).
CONFIG_CLASS = VMPOConfig


def build_policy(obs, action, network, algo):
    from architecture.auxiliary import build_special
    return build_special(obs, action, network, algo, kind='vmpo')

"""Direct advantage estimation: trajectory regression followed by policy improvement."""
from dataclasses import dataclass
import torch as th
from onpolicy.ppo import PPOConfig
from onpolicy._runtime import NativeLoop, scan, run_algorithm

@dataclass
class DAEConfig(PPOConfig):
    critic_epochs: int = 4
    full_action: bool = True
    dae_lambda: float = .95
    normalize_advantage: bool = False
    def __post_init__(self):
        super().__post_init__()
        if self.critic_epochs<1 or not 0<=self.dae_lambda<=1: raise ValueError('Invalid DAE configuration')

def trajectory_residual(rewards, advantages, values, bootstrap, gamma):
    # Differentiable reverse scan: G_t - sum_k gamma^k A(s_{t+k},a_{t+k}) - V_t.
    carry=bootstrap
    seq=[]
    for t in reversed(range(len(rewards))):
        carry=rewards[t]-advantages[t]+gamma*carry
        seq.append(carry-values[t])
    return th.stack(seq[::-1])

class DAE(NativeLoop):
    def _update_(self):
        b=self.rollout_buffer;c=self.config;p=self.policy
        obs=self.flat('obs'); actions=self.flat('action').long()
        with th.no_grad():
            old=p.dist(obs); probs=old.probs.detach(); oldlog=old.logits.detach()
            oldadv=p.advantages(obs,probs).detach()
        # Separate block-coordinate losses: advantage target freezes V; V target freezes A.
        for _ in range(c.critic_epochs):
            losses=[]
            for n,s,e in b.segments():
                idx=th.arange(s,e,device=b.device)*b.N+n
                advantage=p.advantages(obs[idx],probs[idx]).gather(1,actions[idx,None]).squeeze(1)
                value=p.predict_value(obs[idx])
                bootstrap=b['next_value'][e-1,n]*(~b['terminated'][e-1,n])
                a0=oldadv[idx].gather(1,actions[idx,None]).squeeze(1)
                losses.append(trajectory_residual(b['reward'][s:e,n],advantage,b.values[s:e,n],bootstrap,c.gamma).square())
                losses.append(trajectory_residual(b['reward'][s:e,n],a0,value,bootstrap,c.gamma).square())
            critic=th.cat(losses).mean()
            self.step_loss(critic)
        with th.no_grad():
            alladv=p.advantages(obs,probs)
            # Lambda-DAE adds the future sampled advantage to the sampled entry only.
            for n,s,e in b.segments():
                ids=th.arange(s,e,device=b.device)*b.N+n
                taken=alladv[ids,actions[ids]]
                smooth=scan(taken,th.ones_like(taken),c.gamma*c.dae_lambda)
                alladv[ids,actions[ids]]=smooth
            b.advantages.copy_(alladv.gather(1,actions[:,None]).reshape(b.T,b.N))
        for idx in self.batches():
            dist=p.dist(obs[idx]); adv=alladv[idx]; logp=dist.logits
            if c.full_action:
                chosen=th.nn.functional.one_hot(actions[idx],adv.shape[1]).bool()
                logp=th.where(chosen,logp,logp.detach())
                if c.normalize_advantage: adv=adv/((probs[idx]*adv.square()).sum(-1,keepdim=True).sqrt()+1e-6)
                ratio=(logp-oldlog[idx]).exp()
                pg=-(probs[idx]*th.minimum(adv*ratio,adv*ratio.clamp(1-c.epsilon,1+c.epsilon))).sum(-1).mean()
            else:
                adv=adv.gather(1,actions[idx,None]).squeeze(1)
                if c.normalize_advantage: adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
                ratio=(dist.log_prob(actions[idx])-self.flat('logprob')[idx]).exp()
                pg=-th.minimum(adv*ratio,adv*ratio.clamp(1-c.epsilon,1+c.epsilon)).mean()
            kl=(probs[idx]*(oldlog[idx]-dist.logits)).sum(-1).mean()
            loss=pg-c.ent_coef*dist.entropy().mean()+c.kl_coef*kl
            self.step_loss(loss)
            stats={'loss/policy':float(pg.detach()),'loss/dae':float(critic.detach()),'train/kl':float(kl.detach())}
            self.log.add(**stats)
        return stats

def train(cfg,env,policy,log=None,monitor=None,resume=None):
    return run_algorithm(DAE,cfg,env,policy,log,monitor,resume)


# Algorithm discovery contract (no central name/class mapping).
CONFIG_CLASS = DAEConfig


def build_policy(obs, action, network, algo):
    from architecture.auxiliary import build_special
    return build_special(obs, action, network, algo, kind='dae')

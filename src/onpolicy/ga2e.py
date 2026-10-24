"""Gradient-alignment lambda search, rewritten for native tensor rollouts."""
from dataclasses import dataclass
import torch as th
from onpolicy.ppo import PPOConfig
from onpolicy._runtime import NativeLoop, estimates, run_algorithm

@dataclass
class GA2EConfig(PPOConfig):
    lambda_min: float = 0.1
    lambda_val: float = 0.97
    level2_points: int = 10
    val_every: int = 5
    ema_beta: float = 0.9
    two_level: bool = True
    lambda_refresh: str = 'rollout'
    grad_chunk_size: int = 2048
    value_ref_lambda: float | None = None
    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.lambda_min <= 1 and 0 <= self.lambda_val <= 1 and 0 <= self.ema_beta < 1): raise ValueError('Invalid GA2E lambda/EMA range')
        if self.val_every<2 or self.level2_points<1: raise ValueError('Invalid GA2E search parameters')
        if self.lambda_refresh not in ('rollout','epoch'): raise ValueError('Unknown lambda_refresh')

class GA2E(NativeLoop):
    def split(self):
        b=self.rollout_buffer
        train=[]; val=[]
        for i,(n,s,e) in enumerate(b.segments()):
            target=val if i % self.config.val_every == 0 else train
            target.extend(t*b.N+n for t in range(s,e))
        device=b.device
        train=th.tensor(train,dtype=th.long,device=device); val=th.tensor(val,dtype=th.long,device=device)
        degenerate=min(len(train),len(val))<8
        if degenerate: train=val=self.indices()
        self.log.record('ga2e/split_degenerate',float(degenerate))
        return train,val

    def gradient(self, adv, idx, normalize=False):
        a=adv.flatten()[idx].detach()
        if normalize and len(a)>1: a=(a-a.mean())/(a.std(unbiased=False)+1e-8)
        params=list(self.policy.parameters())
        total=[th.zeros_like(p) for p in params]
        chunk=self.config.grad_chunk_size or len(idx)
        eps=self.policy_loss._current_eps(min(1.,self.global_step/self.config.total_steps))
        for start in range(0,len(idx),chunk):
            ids=idx[start:start+chunk]; weights=a[start:start+chunk]
            logp,_,_=self.policy.evaluate(self.flat('obs')[ids],self.flat('action')[ids])
            ratio=(logp-self.flat('logprob')[ids]).exp()
            per=th.minimum(weights*ratio,weights*ratio.clamp(1-eps,1+eps))
            grads=th.autograd.grad(-per.sum()/len(idx),params,allow_unused=True)
            for dest,g in zip(total,grads):
                if g is not None: dest.add_(g.detach())
        return th.cat([g.flatten() for g in total])

    def _compute_advantages_(self):
        b=self.rollout_buffer; c=self.config
        tr,val=self.split()
        reference=self.gradient(estimates(b,c.gamma,c.lambda_val)[0],val)
        cache={}
        def score(lam):
            key=round(float(lam),6)
            if key not in cache:
                grad=self.gradient(estimates(b,c.gamma,key)[0],tr,True)
                cache[key]=float(th.dot(grad,reference)/(grad.norm()*reference.norm()+1e-8))
            return cache[key]
        grid=[max(c.lambda_min,i/10) for i in range(11) if i/10>=c.lambda_min]
        if not grid: grid=[c.lambda_min]
        best=max(grid,key=score)
        if c.two_level:
            lo=max(c.lambda_min,best-.1); hi=min(1.,best+.1)
            best=max([lo+(hi-lo)*i/c.level2_points for i in range(c.level2_points+1)],key=score)
        with th.no_grad():
            if bool(self.policy.lambda_initialized): self.policy.lambda_ema.mul_(c.ema_beta).add_(best*(1-c.ema_beta))
            else: self.policy.lambda_ema.fill_(best); self.policy.lambda_initialized.fill_(True)
        used=float(self.policy.lambda_ema)
        adv,ret=estimates(b,c.gamma,used)
        if c.value_ref_lambda is not None: ret=estimates(b,c.gamma,max(.9,c.value_ref_lambda))[1]
        b.set_estimates(adv,ret)
        self.log.record('ga2e/lambda_star',best); self.log.record('ga2e/lambda_used',used)

    def _update_(self):
        if self.config.lambda_refresh=='rollout': return super()._update_()
        c=self.config; epochs=c.ppo_epoch; seed=c.seed
        fixed_returns=self.rollout_buffer.returns.clone()
        try:
            c.ppo_epoch=1
            for epoch in range(epochs):
                c.seed=seed+epoch
                if epoch:
                    self._compute_advantages_()
                    self.rollout_buffer.returns.copy_(fixed_returns)
                stats=super()._update_()
            return stats
        finally: c.ppo_epoch=epochs; c.seed=seed

def train(cfg,env,policy,log=None,monitor=None,resume=None):
    return run_algorithm(GA2E,cfg,env,policy,log,monitor,resume)


# Algorithm discovery contract (no central name/class mapping).
CONFIG_CLASS = GA2EConfig


def build_policy(obs, action, network, algo):
    from architecture.auxiliary import build_special
    return build_special(obs, action, network, algo, kind='ga2e')

"""RVL one-step branch audited against Hauf3n/relative-value-learning @98ff50b8.

Official terminal-target heuristic is preserved explicitly, including the
both-terminal zero target. No absolute-value critic is trained.
"""
from dataclasses import dataclass
import copy
import torch as th
from onpolicy.ppo import PPOConfig
from onpolicy._runtime import NativeLoop, estimates, run_algorithm

@dataclass
class RVLConfig(PPOConfig):
    rv_coef: float = 1.25
    rv_clip: float | None = .15
    optimal_offset: bool = True
    value_head: str = 'linear'
    same_episode_probability: float = .33
    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.same_episode_probability <= 1: raise ValueError('Invalid pair sampling probability')

def pair_target(ri,rj,delta_next,delta_i_next_j,delta_next_i_j,done_i,done_j,gamma):
    ordinary=ri-rj+gamma*delta_next
    terminal_i=ri-rj+gamma*(delta_i_next_j-ri)
    terminal_j=ri-rj+gamma*(delta_next_i_j+rj)
    return th.where(done_i & done_j,th.zeros_like(ordinary),
                    th.where(done_i,terminal_i,th.where(done_j,terminal_j,ordinary)))

class RVL(NativeLoop):
    @th.no_grad()
    def _compute_advantages_(self):
        b=self.rollout_buffer;p=self.policy;c=self.config
        segs=b.segments();starts=th.stack([b.obs[s,n] for n,s,e in segs])
        offsets=th.zeros(len(segs),device=b.device)
        if c.optimal_offset:
            for i in range(len(segs)):
                offsets[i]=p.relative_value(starts[i].expand_as(starts),starts).mean()
            offsets-=offsets.min()
        values=th.zeros_like(b.values);nxt=th.zeros_like(values)
        for k,(n,s,e) in enumerate(segs):
            delta=p.relative_value(b['next_obs'][s:e,n],b.obs[s:e,n])
            cumulative=th.cat([delta.new_zeros(1),delta.cumsum(0)])+offsets[k]
            values[s:e,n]=cumulative[:-1]
            nxt[s:e,n]=cumulative[1:]
        b.set_estimates(*estimates(b,c.gamma,c.gae_lambda,values,nxt))

    def _update_(self):
        c=self.config;p=self.policy;b=self.rollout_buffer
        old=copy.deepcopy(p).eval()
        for par in old.parameters():par.requires_grad_(False)
        obs=self.flat('obs');nxt=self.flat('next_obs');reward=self.flat('reward');term=self.flat('terminated').bool()
        # Pack complete segments for global/same-episode pair sampling, excluding self.
        segments=[th.arange(s,e,device=b.device)*b.N+n for n,s,e in b.segments()]
        packed=th.cat(segments);inverse=th.empty(b.T*b.N,dtype=th.long,device=b.device)
        inverse[packed]=th.arange(len(packed),device=b.device)
        starts=th.empty_like(inverse);lengths=th.empty_like(inverse);offset=0
        for segment in segments:
            starts[segment]=offset;lengths[segment]=len(segment);offset+=len(segment)
        if len(packed)<2:raise ValueError('RVL needs at least two valid transitions')
        alladv=self.flat('advantages').clone()
        if c.normalize_advantage:
            valid=alladv[packed];alladv=(alladv-valid.mean())/(valid.std(unbiased=False)+1e-8)
        for idx in self.batches():
            anchor=inverse[idx]
            pick=th.randint(len(packed)-1,(len(idx),),device=b.device)
            global_pair=pick+(pick>=anchor)
            length=lengths[idx];start=starts[idx]
            within=1+(th.rand(len(idx),device=b.device)*(length-1).clamp_min(1)).long()
            same=start+(anchor-start+within)%length
            choose=(th.rand(len(idx),device=b.device)<c.same_episode_probability)&(length>1)
            other=packed[th.where(choose,same,global_pair)]
            with th.no_grad():
                previous=old.relative_value(obs[idx],obs[other])
                target=pair_target(reward[idx],reward[other],old.relative_value(nxt[idx],nxt[other]),
                    old.relative_value(obs[idx],nxt[other]),old.relative_value(nxt[idx],obs[other]),term[idx],term[other],c.gamma)
            predicted=p.relative_value(obs[idx],obs[other])
            errors=(predicted-target).square()
            if c.rv_clip is not None:
                clipped=previous+(predicted-previous).clamp(-c.rv_clip,c.rv_clip)
                errors=th.maximum(errors,(clipped-target).square())
            rv=errors.mean()
            logp,entropy,_=p.evaluate(obs[idx],self.flat('action')[idx]);adv=alladv[idx]
            ratio=(logp-self.flat('logprob')[idx]).exp()
            pg=-th.minimum(ratio*adv,ratio.clamp(1-c.epsilon,1+c.epsilon)*adv).mean()
            loss=pg+c.rv_coef*rv-c.ent_coef*entropy.mean()
            self.step_loss(loss)
            stats={'loss/policy':float(pg.detach()),'loss/relative_value':float(rv.detach())}
            self.log.add(**stats)
        return stats

def train(cfg,env,policy,log=None,monitor=None,resume=None):
    return run_algorithm(RVL,cfg,env,policy,log,monitor,resume)


# Algorithm discovery contract (no central name/class mapping).
CONFIG_CLASS = RVLConfig


def build_policy(obs, action, network, algo):
    from architecture.auxiliary import build_special
    return build_special(obs, action, network, algo, kind='rvl')

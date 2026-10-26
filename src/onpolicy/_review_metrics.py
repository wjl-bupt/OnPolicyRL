"""Read-only review diagnostics. No global RNG draws or optimizer changes."""
from dataclasses import dataclass
from common.metrics import explained_variance
import time
import torch as th
from torch.distributions import Normal, Categorical, kl_divergence


@dataclass
class ReviewConfig:
    review_metrics: bool = True
    review_interval: int = 1
    review_max_states: int = 256
    review_actions: int = 8
    review_gradient_interval: int = 0
    review_reference_interval: int = 0
    review_reference_resamples: int = 4
    review_used_lambda: bool = False

    def __post_init__(self):
        super().__post_init__()
        for key in ('review_interval','review_max_states','review_actions','review_reference_resamples'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError(key+' must be a positive integer')
        for key in ('review_gradient_interval','review_reference_interval'):
            if type(getattr(self,key)) is not int or getattr(self,key)<0:raise ValueError(key+' must be a nonnegative integer')


def frozen_dist(d):
    if isinstance(d,Normal):return Normal(d.loc.detach().clone(),d.scale.detach().clone())
    if isinstance(d,Categorical):return Categorical(logits=d.logits.detach().clone())
    raise TypeError('Review metrics require Normal or Categorical policies')


def joint_kl(old,new):
    k=kl_divergence(old,new)
    return k.sum(-1) if isinstance(old,Normal) else k


def sample_independent(d,k,seed):
    device=d.loc.device if isinstance(d,Normal) else d.logits.device
    gen=th.Generator(device=device).manual_seed(seed)
    if isinstance(d,Normal):
        return d.loc+ d.scale*th.randn((k,)+d.loc.shape,device=device,generator=gen)
    return th.multinomial(d.probs,k,replacement=True,generator=gen).T


def describe(log,prefix,x):
    x=x.detach().float().flatten()
    log.record(prefix+'/nonfinite_fraction',float((~th.isfinite(x)).float().mean()))
    x=x[th.isfinite(x)]
    if not x.numel():return
    log.record(prefix+'/mean',float(x.mean()))
    log.record(prefix+'/std',float(x.std(unbiased=False)))
    log.record(prefix+'/max',float(x.max()))
    for q,v in zip((5,50,95,99),th.quantile(x,th.tensor([.05,.5,.95,.99],device=x.device))):log.record(prefix+'/p'+str(q),float(v))


def ratios(log,prefix,logratio,eps):
    describe(log,prefix+'/ratio',logratio.exp())
    describe(log,prefix+'/abs_log_ratio',logratio.abs())
    log.record(prefix+'/outside_clip',float(((logratio.exp()-1).abs()>eps).float().mean()))


def actor_mask(policy):
    modules=[policy.policy_head]
    modules += [policy.extractor,policy.pi_post] if policy.share else [policy.pi_extractor]
    ids={id(p) for m in modules for p in m.parameters()}
    return th.cat([th.full((p.numel(),),id(p) in ids,device=p.device,dtype=th.bool) for p in policy.parameters()])


def vector(policy):return th.cat([p.detach().reshape(-1) for p in policy.parameters()])


def alignment(log,prefix,a,b):
    a=a.detach().double();b=b.detach().double();den=a.norm()*b.norm()
    log.record(prefix+'/dot',float(a@b))
    log.record(prefix+'/cosine',float((a@b)/den) if den>1e-12 else 0.)
    log.record(prefix+'/near_zero',float(den<=1e-12))


class ReviewMixin:
    def train(self) -> None:
        """The outer loop: collect -> estimate -> update -> report."""
        import time
        session_start=time.perf_counter();session_step=self.global_step
        if self.monitor is not None:
            self.monitor._session_clock=session_start;self.monitor._session_step=session_step
        while self.global_step < self.config.total_steps:
            self._collect_rollout_()
            if self.config.anneal_lr:
                progress = min(1.0, self.global_step / self.config.total_steps)
                for group in self.policy.optimizer.param_groups:
                    initial = group.setdefault("initial_lr", group["lr"])
                    group["lr"] = self.config.lr_min + (initial-self.config.lr_min)*(1-progress)
            self._compute_advantages_()
            stats = self._update_()
            self.num_updates += 1

            # critic health -- an underrated diagnostic (metrics.py)
            if stats:
                buf = self.rollout_buffer
                self.log.record("train/explained_variance",
                                explained_variance(buf.values, buf.returns))
            self.log.record("perf/sps", self._sps_and_drain())
            self.log.record("train/updates", self.num_updates)
            if self.monitor is not None:               # report-only contact point
                self.monitor.observe(stats, global_step=self.global_step,
                                     iteration=self.num_updates)
            elapsed=time.perf_counter()-session_start
            self.log.record('perf/session_wall_seconds',elapsed)
            self.log.record('perf/session_steps',self.global_step-session_step)
            self.log.record('perf/end_to_end_sps',(self.global_step-session_step)/max(elapsed,1e-9))
            if self.num_updates % self.config.log_interval == 0:
                self.log.dump(self.global_step)

        self.log.dump(self.global_step)


    def set_review_probe(self, observations, actions, advantages, *, source, seed):
        """Attach a fresh externally collected independent batch for ONE update.

        Caller must collect it with the current policy on independent trajectories;
        this API neither collects extra interactions nor uses it for optimization.
        """
        if not source:raise ValueError('Independent probe requires a source identifier')
        if len(observations)!=len(actions) or len(actions)!=len(advantages) or not len(actions):
            raise ValueError('Probe arrays must have equal nonzero length')
        self._review_probe=(observations.detach().clone(),actions.detach().clone(),advantages.detach().clone(),str(source),int(seed))

    def _review_due(self):return self.config.review_metrics and self.num_updates%self.config.review_interval==0

    def _collect_rollout_(self):
        start=time.perf_counter();result=super()._collect_rollout_()
        self.log.record('review_time/rollout_seconds',time.perf_counter()-start)
        return result

    def _compute_advantages_(self):
        start=time.perf_counter();result=super()._compute_advantages_()
        self.log.record('review_time/estimation_seconds',time.perf_counter()-start)
        if self._review_due():
            ids=self.indices();b=self.rollout_buffer
            describe(self.log,'review/advantage',self.flat('advantages')[ids])
            for name in ('terminated','truncated'):self.log.record('review/'+name+'_fraction',float(self.flat(name)[ids].float().mean()))
            self.log.record('review/valid_samples',len(ids))
        return result

    @th.no_grad()
    def _review_snapshot(self):
        ids=self.indices();n=min(len(ids),self.config.review_max_states)
        ids=ids[th.linspace(0,len(ids)-1,n,device=ids.device).long()]
        obs=self.flat('obs')[ids];old=frozen_dist(self.policy.dist(obs))
        anchors=sample_independent(old,self.config.review_actions,self.config.seed+7919*(self.num_updates+1))
        params=vector(self.policy).clone()
        return ids,obs,old,anchors,params

    @th.no_grad()
    def _review_after(self,snapshot):
        ids,obs,old,anchors,before=snapshot;p=self.policy;log=self.log
        new=p.dist(obs);eps=self.policy_loss._current_eps(min(1.,self.global_step/self.config.total_steps))
        describe(log,'review/analytic_kl',joint_kl(old,new))
        describe(log,'review/entropy',p.policy_head.entropy(new))
        if isinstance(new,Normal):describe(log,'review/policy_std',new.scale)
        acts=self.flat('action')[ids]
        ratios(log,'review/executed',p.policy_head.log_prob(new,acts)-self.flat('logprob')[ids],eps)
        ratios(log,'review/independent',p.policy_head.log_prob(new,anchors)-p.policy_head.log_prob(old,anchors),eps)
        log.record('review/diagnostic_states',len(ids));log.record('review/independent_actions_per_state',self.config.review_actions)
        if 'uact' in self.rollout_buffer:
            ua=self.flat('uact')[ids].movedim(0,1)
            oldlp=self.flat('ua_logprobs')[ids].T
            lr=p.policy_head.log_prob(new,ua)-oldlp
            ratios(log,'review/anchors',lr,eps)
            describe(log,'review/anchor_old_log_density',oldlp)
            weight=oldlp.exp() if self.config.uarr_coef<0 else th.full_like(oldlp,self.config.uarr_coef)
            describe(log,'review/anchor_weight',weight)
            describe(log,'review/uarr_unweighted',.5*(lr.exp()-1).square())
            describe(log,'review/uarr_weighted',.5*weight*(lr.exp()-1).square())
            log.record('review/ua_nums',self.config.ua_nums)
        mask=actor_mask(p);delta=(vector(p)-before)[mask]
        log.record('review/actor_parameter_delta_norm',float(delta.norm()))
        # Search code computes LOSS gradients: negate to obtain ascent directions.
        for name in ('reference','selected'):
            g=getattr(self,'_review_update_'+name,None)
            if g is not None:alignment(log,'review/update_vs_'+name,delta,-g[mask])

    def _update_(self):
        due=self._review_due();diag=0.;start=time.perf_counter()
        probe=getattr(self,'_review_probe',None);self._review_probe=None
        if probe is not None:
            with th.no_grad():probe_old=self.policy.evaluate(probe[0],probe[1])[0].detach().clone()
        if due:
            snapshot=self._review_snapshot();diag+=time.perf_counter()-start
            for name in ('reference','selected'):
                setattr(self,'_review_update_'+name,getattr(self,'_review_'+name,None))
        steps=0
        def count(*args):
            nonlocal steps
            steps+=1
        handle=self.policy.optimizer.register_step_post_hook(count)
        try:result=super()._update_()
        finally:handle.remove()
        update_elapsed=time.perf_counter()-start-diag
        if due:
            start=time.perf_counter();self._review_after(snapshot);diag+=time.perf_counter()-start
        if probe is not None:
            start=time.perf_counter()
            with th.no_grad():
                ratio=(self.policy.evaluate(probe[0],probe[1])[0]-probe_old).exp()
                eps=self.policy_loss._current_eps(min(1.,self.global_step/self.config.total_steps))
                gain=(th.min(ratio*probe[2],ratio.clamp(1-eps,1+eps)*probe[2])-probe[2]).mean()
            self.log.record('review_probe/clipped_surrogate_gain',float(gain))
            self.log.record('review_probe/seed',probe[4]);self.log.record('review_probe/samples',len(probe[1]))
            diag+=time.perf_counter()-start
        if not hasattr(self, '_first_selection'):
            batches=min(self.config.num_mini_batch,len(self.indices()))
            self.log.record('review/epochs_entered',(steps+batches-1)//max(1,batches))
            threshold=self.config.target_kl
            self.log.record('review/early_stop',float(threshold is not None and result.get('train/kl',0)>threshold))
        self.log.record('review/optimizer_steps',steps)
        self.log.record('review_time/update_seconds',update_elapsed)
        self.log.record('review_time/diagnostic_seconds',diag)
        if self.rollout_buffer.device.type=='cuda':self.log.record('review/peak_cuda_bytes',th.cuda.max_memory_allocated(self.rollout_buffer.device))
        return result

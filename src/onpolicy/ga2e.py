"""Native GA2E aligned to Posterior-GAE 17ff77b73de240974d42a1e3c3046da16e708c8c.

No SB3 runtime dependency. Explicitly preserves the source's extra epoch-zero
selection, NumPy search reductions, two different minibatch layouts, and losses.
"""
from dataclasses import dataclass
import time
from onpolicy._review_metrics import ReviewConfig, ReviewMixin
from onpolicy._ga2e_review import search_review, split_review, ema_review
import numpy as np
import torch as th
from torch.nn import functional as F
from onpolicy.ppo import PPOConfig
from onpolicy._runtime import NativeLoop, run_algorithm


@dataclass
class GA2EConfig(ReviewConfig, PPOConfig):
    update_data: str = 'all'  # actor: all | candidate; critic ALWAYS uses all
    lambda_min: float = .1
    lambda_val: float = .97
    level2_points: int = 10
    val_every: int = 5
    ema_beta: float = .9
    two_level: bool = True
    lambda_refresh: str = 'rollout'
    var_penalty_beta: float = 0.
    n_var_folds: int = 3
    grad_chunk_size: int = 0
    use_spo: bool = False
    value_ref_lambda: float | None = None
    batch_size: int | None = None  # explicit SB3 batch_size; None derives B/num_mini_batch
    value_loss_scale: float = 1.0
    parity_version: str = 'posterior-17ff77b-v1'

    def __post_init__(self):
        super().__post_init__()
        if self.update_data not in ("all","candidate"):raise ValueError("update_data must be all or candidate")
        if not (0<=self.lambda_min<=1 and 0<=self.lambda_val<=1 and 0<=self.ema_beta<1):
            raise ValueError('Invalid GA2E lambda/EMA range')
        if self.val_every<2 or self.level2_points<1:raise ValueError('Invalid GA2E search parameters')
        if self.lambda_refresh not in ('rollout','epoch'):raise ValueError('Unknown lambda_refresh')
        if self.batch_size is not None and (type(self.batch_size) is not int or self.batch_size<1):raise ValueError('Invalid batch_size')
        if self.value_loss_scale != 1.:raise ValueError('Reference GA2E uses unscaled MSE: value_loss_scale must be 1')
        if self.value_ref_lambda is not None:self.value_ref_lambda=max(.9,float(self.value_ref_lambda))
        if self.parity_version != 'posterior-17ff77b-v1':raise ValueError('Unsupported GA2E parity version')


class _GA2ECore(NativeLoop):
    def _deltas_cont(self):
        """SB3-equivalent timeout reward correction; SameStep has all valid rows.

        At epoch boundaries only the final bootstrap is re-evaluated, just as
        the reference. Interior values remain rollout values.
        """
        b,c=self.rollout_buffer,self.config
        values=b.values.detach().cpu().numpy()
        reward=b['reward'][:b.T].detach().cpu().numpy().copy()
        terminal=b['terminated'][:b.T].cpu().numpy().astype(bool)
        truncated=b['truncated'][:b.T].cpu().numpy().astype(bool)
        valid=b['valid'][:b.T].cpu().numpy().astype(bool)
        next_value=b['next_value'][:b.T].detach().cpu().numpy()
        # SB3 adds discounted terminal-observation value to timeout reward.
        reward += c.gamma*next_value*(truncated & ~terminal)
        cont=(~(terminal|truncated)&valid).astype(np.float32)
        nxt=np.empty_like(values);nxt[:-1]=values[1:]
        with th.no_grad():nxt[-1]=self.policy.predict_value(self._last_obs).cpu().numpy().flatten()
        # Extension only for adapters with dummy NextStep resets (SB3 has none).
        for t in range(b.T-1):
            missing=~valid[t+1]
            nxt[t,missing]=next_value[t,missing]
            cont[t,missing]=0.
        delta=(reward+c.gamma*nxt*cont-values).astype(np.float32)
        delta[~valid]=0.
        return delta,cont

    def _gae(self,delta,cont,lam):
        adv=np.zeros_like(delta,dtype=np.float32);carry=np.zeros(delta.shape[1],dtype=np.float32)
        gl=self.config.gamma*lam
        for t in reversed(range(len(delta))):
            carry=delta[t]+gl*cont[t]*carry;adv[t]=carry
        return adv

    def split(self):
        b=self.rollout_buffer;segments=b.segments()
        tr=np.zeros((b.T,b.N),dtype=bool);val=np.zeros_like(tr)
        for i,(n,s,e) in enumerate(segments):
            (val if i%self.config.val_every==0 else tr)[s:e,n]=True
        # Search uses C-order (time-major), NOT concatenated environment segments.
        train,validation=np.flatnonzero(tr.reshape(-1)),np.flatnonzero(val.reshape(-1))
        self._candidate_indices=train.copy();self._reference_indices=validation.copy()
        self._n_segments=len(segments);self._val_frac=len(validation)/(b.T*b.N)
        self._split_degenerate=min(len(train),len(validation))<8
        if self._split_degenerate:
            train=validation=np.flatnonzero(b['valid'][:b.T].cpu().numpy().reshape(-1));self._val_frac=1.
        if self.config.review_metrics:split_review(self,segments,train,validation)
        return train,validation

    def _policy_grads_batch(self,advantages,idx,normalize=False,clip_range=None):
        params=list(self.policy.parameters());idx=np.asarray(idx);m=len(idx)
        if not m:raise ValueError('No valid GA2E search samples')
        device=self.rollout_buffer.device;adv_ts=[]
        for a in advantages:
            a=a.reshape(-1)[idx]
            if normalize and a.size>1:a=(a-a.mean())/(a.std()+1e-8)
            adv_ts.append(th.as_tensor(a,dtype=th.float32,device=device))
        idx=th.as_tensor(idx,dtype=th.long,device=device)
        cs=self.config.grad_chunk_size;single=cs<=0 or m<=cs;step=m if single else cs
        sizes=[p.numel() for p in params];offsets=np.cumsum([0,*sizes[:-1]])
        out=th.zeros((len(advantages),sum(sizes)),device=device)
        for start in range(0,m,step):
            sl=slice(start,min(start+step,m));ids=idx[sl]
            logp,_,_=self.policy.evaluate(self.flat('obs')[ids],self.flat('action')[ids])
            ratio=(logp-self.flat('logprob')[ids]).exp() if clip_range is not None else None
            for j,a in enumerate(adv_ts):
                per=logp*a[sl] if clip_range is None else th.min(a[sl]*ratio,a[sl]*ratio.clamp(1-clip_range,1+clip_range))
                self.policy.zero_grad(set_to_none=False)
                (-per.mean() if single else -per.sum()/m).backward(retain_graph=j+1<len(adv_ts))
                for p,offset,size in zip(params,offsets,sizes):
                    if p.grad is not None:out[j,int(offset):int(offset)+size]+=p.grad.reshape(-1)
        self.policy.zero_grad(set_to_none=True)
        return out

    def gradient(self,adv,idx,normalize=False,clip_range=None):
        if isinstance(adv,th.Tensor):adv=adv.detach().cpu().numpy()
        if isinstance(idx,th.Tensor):idx=idx.cpu().numpy()
        # Matches source's single-gradient accumulation (not batched sum reduction).
        a=adv.reshape(-1)[idx]
        if normalize and a.size>1:a=(a-a.mean())/(a.std()+1e-8)
        device=self.rollout_buffer.device;a=th.as_tensor(a,dtype=th.float32,device=device)
        ids=th.as_tensor(idx,dtype=th.long,device=device);m=len(ids);cs=self.config.grad_chunk_size
        if not m:raise ValueError('No valid GA2E search samples')
        single=cs<=0 or m<=cs;step=m if single else cs
        self.policy.zero_grad(set_to_none=True)
        for start in range(0,m,step):
            part=ids[start:start+step];weights=a[start:start+step]
            logp,_,_=self.policy.evaluate(self.flat('obs')[part],self.flat('action')[part])
            if clip_range is None:per=logp*weights
            else:
                ratio=(logp-self.flat('logprob')[part]).exp()
                per=th.min(weights*ratio,weights*ratio.clamp(1-clip_range,1+clip_range))
            (-per.mean() if single else -per.sum()/m).backward()
        result=th.cat([p.grad.detach().reshape(-1) if p.grad is not None else th.zeros(p.numel(),device=device) for p in self.policy.parameters()])
        self.policy.zero_grad(set_to_none=True)
        return result

    def _select_lambda(self,clip_range=None):
        search_start=time.perf_counter()
        c=self.config;delta,cont=self._deltas_cont();tr,val=self.split()
        ref=self.gradient(self._gae(delta,cont,c.lambda_val),val,clip_range=clip_range)
        ref_raw=ref.detach().clone();diagnostic_grads={};gradient_evals=1
        ref=ref/(ref.norm()+1e-8);cache={};parts={};grad_cache={}
        def evaluate(grid):
            nonlocal gradient_evals
            if c.var_penalty_beta<=0:
                todo=[lam for lam in grid if round(float(lam),3) not in grad_cache]
                if todo:
                    gradient_evals+=len(todo)
                    gradients=self._policy_grads_batch([self._gae(delta,cont,l) for l in todo],tr,True,clip_range)
                    for lam,g in zip(todo,gradients):grad_cache[round(float(lam),3)]=g.clone()
                for lam in grid:
                    key=round(float(lam),3);g=grad_cache[key]
                    bias=float(ref@g/(g.norm()+1e-8));cache[key]=bias;parts[key]=(bias,0.)
                    diagnostic_grads[key]=g
            else:
                for lam in grid:
                    key=round(float(lam),3)
                    if key in cache:continue
                    adv=self._gae(delta,cont,key)
                    gs=[self.gradient(adv,f,True,clip_range) for f in np.array_split(tr,max(2,c.n_var_folds)) if f.size>=4]
                    gradient_evals+=len(gs)
                    G=th.stack(gs);mean=G.mean(0)
                    diagnostic_grads[key]=mean
                    bias=float(ref@mean/(mean.norm()+1e-8))
                    var=float(((G-mean)**2).sum(1).mean())/(float((mean**2).sum())+1e-12)
                    cache[key]=bias-c.var_penalty_beta*var;parts[key]=(bias,var)
        grid=[round(.1*i,2) for i in range(int(round(c.lambda_min*10)),11)]
        evaluate(grid);best=max(grid,key=lambda l:cache[round(float(l),3)])
        if c.two_level:
            lo=max(c.lambda_min,round(best-.1,3));hi=min(1.,round(best+.1,3))
            grid=[round(lo+(hi-lo)*j/c.level2_points,3) for j in range(c.level2_points+1)]
            evaluate(grid);best=max(grid,key=lambda l:cache[round(float(l),3)])
        self._score_bias,self._score_var=parts[round(best,3)]
        self.log.record('review_time/search_seconds',time.perf_counter()-search_start)
        self.log.record('review_search/gradient_evaluations',gradient_evals)
        self.log.record('review_search/search_calls',1)
        if c.review_metrics:
            search_review(self,best,cache,parts,diagnostic_grads,ref_raw,delta,cont,tr,val,clip_range)
        return best,cache

    def _ema_update(self,star):
        p=self.policy;c=self.config
        previous=float(p.lambda_ema)
        raw=c.ema_beta*previous+(1-c.ema_beta)*star if bool(p.lambda_initialized) else star
        p.lambda_ema.fill_(raw);p.lambda_initialized.fill_(True)
        used=float(np.clip(raw,c.lambda_min,1.))
        if c.review_metrics:ema_review(self,star,previous,raw,used)
        return used

    def _record_lambda(self,star,used,curve,n_evals=None):
        values={'lambda_star':star,'lambda_used':used,'val_frac':self._val_frac,
                'train_frac':1. if self._split_degenerate else 1-self._val_frac,
                'n_segments':self._n_segments,'split_degenerate':float(self._split_degenerate)}
        if curve:values.update(score_best=max(curve.values()),n_evals=len(curve) if n_evals is None else n_evals)
        if self.config.var_penalty_beta>0:values.update(score_bias=self._score_bias,score_var=self._score_var)
        for key,value in values.items():self.log.record('ga2e/'+key,float(value))

    def _compute_advantages_(self):
        self.policy.train();c=self.config;b=self.rollout_buffer
        # Fixed epoch training targets; lambda-search bootstrap may change later.
        self._training_delta,self._training_cont=self._deltas_cont()
        eps=self.policy_loss._current_eps(min(1.,self.global_step/c.total_steps))
        star,curve=self._select_lambda(eps if c.lambda_refresh=='epoch' else None)
        used=self._ema_update(star);self._first_selection=(star,used,curve)
        adv=self._gae(self._training_delta,self._training_cont,used)
        ref=c.value_ref_lambda
        returns_adv=adv
        if ref is not None and (c.lambda_refresh=='epoch' or abs(ref-used)>1e-6):
            returns_adv=self._gae(self._training_delta,self._training_cont,ref)
        ret=returns_adv+b.values.detach().cpu().numpy()
        b.set_estimates(th.as_tensor(adv,device=b.device),th.as_tensor(ret,device=b.device))
        self._record_lambda(star,used,curve)

    def _update_(self):
        c=self.config;b=self.rollout_buffer;epoch_mode=c.lambda_refresh=='epoch'
        star0,used0,curve0=self._first_selection;history=[used0];evaluations=len(curve0)
        valid=self.indices().cpu().numpy();N=b.T*b.N
        # SB3 rollout buffer get() swaps T/E before flattening. Custom epoch mode does not.
        order=np.arange(N).reshape(b.T,b.N).T.reshape(-1) if not epoch_mode else np.arange(N)
        order=order[np.isin(order,valid)]
        bs=c.batch_size or max(1,len(order)//c.num_mini_batch)
        # Keep all minibatches for the critic; mask only the actor objective.
        if c.update_data == 'candidate':
            if not len(self._candidate_indices):raise ValueError('candidate-only actor has no candidate samples; collect more trajectory segments')
        self._critic_update_indices=order.copy()
        self._update_indices=order[np.isin(order,self._candidate_indices)] if c.update_data=='candidate' else order.copy()
        self.log.record('review/actor_update_samples',len(self._update_indices))
        self.log.record('review/critic_update_samples',len(order))
        self.log.record('review/update_candidate_only',float(c.update_data=='candidate'))
        self.log.record('review/update_reference_overlap',len(np.intersect1d(self._update_indices,self._reference_indices)))
        self.log.record('review/critic_target_lambda',c.value_ref_lambda if c.value_ref_lambda is not None else used0)
        progress=min(1.,self.global_step/c.total_steps);eps=self.policy_loss._current_eps(progress)
        records=[];stop=False;actor_stopped=False
        candidate_mask=th.zeros(N,dtype=th.bool,device=b.device)
        candidate_mask[th.as_tensor(self._candidate_indices,device=b.device)]=True
        for epoch in range(c.ppo_epoch):
            if epoch_mode:
                # Intentionally repeat selection even for epoch 0 (reference's executed code).
                star,curve=self._select_lambda(eps);used=self._ema_update(star)
                history.append(used);evaluations+=len(curve)
                adv=self._gae(self._training_delta,self._training_cont,used)
                b.advantages.copy_(th.as_tensor(adv,device=b.device))
            permutation=np.random.permutation(len(order))
            for start in range(0,len(order),bs):
                ids=th.as_tensor(order[permutation[start:start+bs]],device=b.device)
                with self.timer('bwd'):
                    logp,entropy,value=self.policy.evaluate(self.flat('obs')[ids],self.flat('action')[ids])
                    adv=self.flat('advantages')[ids]
                    old=self.flat('logprob')[ids]
                    if c.update_data=='candidate':
                        mask=candidate_mask[ids] if not actor_stopped else th.zeros_like(ids,dtype=th.bool)
                        logp=logp[mask];adv=adv[mask];old=old[mask]
                        if entropy is not None:entropy=entropy[mask]
                    if c.normalize_advantage and len(adv)>1:adv=(adv-adv.mean())/(adv.std()+1e-8)
                    ratio=(logp-old).exp()
                    spo=c.use_spo and not epoch_mode
                    if not len(adv):pg=value.new_zeros(())
                    elif spo:pg=-(adv*ratio-adv.abs()*(ratio-1).square()/(2*max(1e-3,eps))).mean()
                    else:pg=-th.min(adv*ratio,adv*ratio.clamp(1-eps,1+eps)).mean()
                    pred=value if not c.use_value_clipped else self.flat('value')[ids]+(value-self.flat('value')[ids]).clamp(-c.vf_epsilon,c.vf_epsilon)
                    vl=F.mse_loss(self.flat('returns')[ids],pred)
                    ent_loss=(-entropy.mean() if entropy is not None else logp.mean()) if len(adv) else value.new_zeros(())
                    entropy_coef=max(.01,c.ent_coef*(1-progress)) if spo else c.ent_coef
                    logratio=logp-old;kl=(logratio.exp()-1-logratio).mean() if len(adv) else value.new_zeros(())
                    loss=pg+entropy_coef*ent_loss+c.vf_coef*vl
                    if not epoch_mode and not spo and c.kl_coef>0:loss=loss+c.kl_coef*kl
                    stat={'loss/policy':float(pg.detach()),'loss/value':float(vl.detach()),'loss/entropy':float(ent_loss.detach()),
                        'loss/total':float(loss.detach()),'train/kl':float(kl.detach()),'train/clip_range':eps,
                        'train/clipfrac':float(((ratio-1).abs()>eps).float().mean().detach()) if len(adv) else 0.,
                        'review/actor_minibatch_samples':len(adv),'review/critic_minibatch_samples':len(ids)}
                    records.append(stat)
                    # Reference stops before the offending optimizer step, at 1.5*threshold.
                    if c.target_kl is not None and stat['train/kl']>1.5*c.target_kl:
                        if c.update_data=='all':stop=True;break
                        # Candidate mode still trains value on every valid row.
                        actor_stopped=True;loss=c.vf_coef*vl
                        stat.update({'loss/total':float(loss.detach()),'loss/policy':0.,'loss/entropy':0.,'review/actor_minibatch_samples':0})
                    self.policy.optimizer.zero_grad();loss.backward()
                    norm=th.nn.utils.clip_grad_norm_(self.policy.parameters(),c.max_grad_norm)
                    self.policy.optimizer.step();stat['train/grad_norm']=float(norm)
                self.log.add(**stat)
            if stop:break
        self.log.record('review/epochs_entered',epoch+1)
        self.log.record('review/early_stop',float(stop))
        self.log.record('review/actor_early_stop',float(actor_stopped))
        if epoch_mode:
            self._record_lambda(star0,float(np.mean(history)),curve0,evaluations)
            self.log.record('ga2e/lambda_epoch_std',float(np.std(history)))
        return records[-1] if records else {}


class GA2E(ReviewMixin, _GA2ECore):
    """Wrap original estimation/update entrypoints with read-only diagnostics."""


def train(cfg,env,policy,log=None,monitor=None,resume=None):
    return run_algorithm(GA2E,cfg,env,policy,log,monitor,resume)

CONFIG_CLASS=GA2EConfig


def build_policy(obs,action,network,algo):
    from architecture.auxiliary import build_special
    return build_special(obs,action,network,algo,kind='ga2e')

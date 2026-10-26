"""GA2E diagnostics consume existing search gradients; never select lambda."""
import time
import json
import numpy as np
import torch as th
from onpolicy._review_metrics import describe,alignment


def split_review(a,segments,tr,val):
    log=a.log
    log.record('review_search/reference_samples',len(val));log.record('review_search/candidate_samples',len(tr))
    log.record('review_search/reference_segments',sum(i%a.config.val_every==0 for i in range(len(segments))))
    log.record('review_search/candidate_segments',sum(i%a.config.val_every!=0 for i in range(len(segments))))
    log.record('review_search/overlap_samples',len(np.intersect1d(tr,val)))
    log.record('review_search/split_degenerate',float(a._split_degenerate))
    lengths=th.tensor([e-s for _,s,e in segments],dtype=th.float32)
    if len(lengths):describe(log,'review_search/segment_length',lengths)
    b=a.rollout_buffer
    boundary=sum(e==b.T and not bool(b['terminated'][e-1,n] or b['truncated'][e-1,n]) for n,s,e in segments)
    log.record('review_search/rollout_boundary_segments',boundary)


def search_review(a,best,scores,parts,grads,ref,delta,cont,tr,val,clip):
    start=time.perf_counter();log=a.log;c=a.config
    norm=float(ref.norm());log.record('review_search/reference_norm',norm)
    log.record('review_search/reference_near_zero',float(norm<1e-10))
    log.record('review_search/reference_nonfinite',float(not th.isfinite(ref).all()))
    dots={};cosines={}
    for lam,g in grads.items():
        key=f'review_candidates/lambda_{lam:.3f}'
        dots[lam]=float(ref@g);cosines[lam]=parts[lam][0]
        log.record(key+'_cosine',cosines[lam]);log.record(key+'_dot',dots[lam])
        log.record(key+'_norm',float(g.norm()));log.record(key+'_score',scores[lam])
        log.record(key+'_variance_penalty',parts[lam][1])
    cos_best=max(cosines,key=cosines.get);dot_best=max(dots,key=dots.get)
    ranked=sorted(cosines.values(),reverse=True)
    log.record('review_search/cosine_best_lambda',cos_best);log.record('review_search/dot_best_lambda',dot_best)
    log.record('review_search/selection_disagreement',float(cos_best!=dot_best))
    log.record('review_search/cosine_margin',ranked[0]-ranked[1] if len(ranked)>1 else 0.)
    log.record('review_search/cosine_range',max(ranked)-min(ranked))
    log.record('review_search/candidates',len(grads));log.record('review_search/reference_lambda',c.lambda_val)
    a._review_reference=ref.detach().clone();a._review_selected=grads[round(best,3)].detach().clone()
    a._review_search=(delta,cont,tr,val,clip,grads,ref)
    a._review_trace=dict(step=a.global_step,update=a.num_updates,lambda_star=best,reference_norm=norm,
        reference_lambda=c.lambda_val,update_data=c.update_data,lambda_refresh=c.lambda_refresh,
        candidate_samples=len(tr),reference_samples=len(val),split_degenerate=a._split_degenerate,
        candidates=[dict(lambda_value=k,cosine=cosines[k],dot=dots[k],norm=float(grads[k].norm()),score=scores[k]) for k in grads])
    interval=c.review_reference_interval
    # Whole trajectory resampling, never individual temporally correlated rows.
    if interval and a.num_updates%interval==0:
        segments=[];valset=set(val.tolist());b=a.rollout_buffer
        for n,s,e in b.segments():
            ids=np.arange(s,e)*b.N+n
            if all(int(i) in valset for i in ids):segments.append(ids)
        log.record('review_reference/available_segments',len(segments))
        if len(segments)>=2:
            adv=a._gae(delta,cont,c.lambda_val)
            g1=a.gradient(adv,np.concatenate(segments[::2]),clip_range=clip)
            g2=a.gradient(adv,np.concatenate(segments[1::2]),clip_range=clip)
            alignment(log,'review_reference/split_half',g1,g2)
            rng=np.random.default_rng(c.seed+104729*(a.num_updates+1));winners=[];values=[]
            keys=list(grads)
            for _ in range(c.review_reference_resamples):
                ids=np.concatenate([segments[i] for i in rng.integers(len(segments),size=len(segments))])
                g=a.gradient(adv,ids,clip_range=clip)
                sims=[float(g@grads[k]/(g.norm()*grads[k].norm()+1e-8)) for k in keys]
                winners.append(keys[int(np.argmax(sims))]);values.append(sims)
            log.record('review_reference/winner_agreement',float(np.mean(np.array(winners)==cos_best)))
            log.record('review_reference/mean_score_std',float(np.std(values,axis=0).mean()))
            log.record('review_reference/extra_gradient_evaluations',2+c.review_reference_resamples)
        else:log.record('review_reference/insufficient_segments',1.)
    log.record('review_time/search_diagnostic_seconds',time.perf_counter()-start)


def ema_review(a,star,previous,raw,used):
    log=a.log;c=a.config
    for key,value in dict(raw_lambda=star,ema_before=previous,ema_after=raw,used_lambda=used,ema_beta=c.ema_beta,
                          smoothing_gap=used-star,lambda_change=used-previous,boundary_hit=float(used<=c.lambda_min or used>=1.)).items():
        log.record('review_lambda/'+key,value)
    if hasattr(a,'_review_trace') and a.monitor is not None and hasattr(a.monitor,'run_dir'):
        from pathlib import Path
        directory=Path(a.monitor.run_dir);directory.mkdir(parents=True,exist_ok=True)
        trace={**a._review_trace,'ema_before':previous,'ema_after':raw,'lambda_used':used,'ema_beta':c.ema_beta}
        # One record per actual search, including every epoch-zero search.
        with (directory/'ga2e_search.jsonl').open('a') as f:f.write(json.dumps(trace)+'\n')
    if not hasattr(a,'_review_search'):return
    delta,cont,tr,val,clip,grads,ref=a._review_search
    # Only reuse a gradient when lambda is exactly the evaluated grid point.
    key=next((k for k in grads if abs(k-used)<1e-12),None)
    g=grads[key] if key is not None else None
    if g is None and c.review_used_lambda:
        start=time.perf_counter()
        g=a.gradient(a._gae(delta,cont,used),tr,True,clip)
        log.record('review_time/used_lambda_diagnostic_seconds',time.perf_counter()-start)
        log.record('review_search/extra_used_gradient_evaluations',1)
    log.record('review_lambda/used_score_available',float(g is not None))
    if g is not None:
        alignment(log,'review_lambda/used_gradient',g,ref)
        log.record('review_lambda/used_gradient_norm',float(g.norm()))

"""Offline seed-level reporting from an explicit expected-run manifest.

python -m workflow.review_report manifest.json --output report_dir
Never selects seeds, invents normalization, or extrapolates unfinished runs.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np


def flatten(d,prefix=''):
    result={}
    for k,v in d.items():
        if isinstance(v,dict):result.update(flatten(v,prefix+k+'/'))
        else:result[prefix+k]=v
    return result


def read_curve(path):
    merged={}
    for line in path.read_text().splitlines():
        if not line.strip():continue
        row=flatten(json.loads(line));step=row.get('step',row.get('global_step'))
        if step is not None:merged.setdefault(float(step),{}).update(row)
    return [(s,v) for s,v in sorted(merged.items()) if np.isfinite(v.get('eval/ep_rew_mean',np.nan))]


def ci(values,rng,count=2000):
    if len(values)<2:return None
    return np.quantile(rng.choice(values,(count,len(values)),replace=True).mean(1),[.025,.975]).tolist()


def iqm(values):
    x=np.sort(np.asarray(values,float));n=len(x);e=np.arange(n+1)
    weights=np.maximum(0,np.minimum(e[1:],.75*n)-np.maximum(e[:-1],.25*n))
    return float(x@weights/weights.sum())


def auc(x,y):
    # Compatible with NumPy 1.x and 2.x; domains are always reported.
    return float(np.sum(np.diff(x)*(y[1:]+y[:-1])*.5)) if len(x)>1 else None


def build_report(manifest,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(manifest.get('bootstrap_seed',0));rows=[];curves=[];warnings=[];series={}
    seen=set();norm=manifest.get('normalization',{});tail=float(manifest.get('final_fraction',.1))
    if not 0<tail<=1:raise ValueError('final_fraction must lie in (0,1]')
    for env,n in norm.items():
        if n['high']<=n['low'] or not n.get('source'):raise ValueError('Normalization needs high>low and an explicit source')
    for run in manifest['runs']:
        key=(run['algorithm'],run['environment'],run['seed'])
        if key in seen:raise ValueError('Duplicate algorithm/environment/seed')
        seen.add(key);path=Path(run['path']);budget=float(run['budget_steps'])
        result={**run,'status':'missing','final_score':None};rows.append(result)
        f=path/'metrics.jsonl'
        if not f.exists():continue
        points=read_curve(f)
        if not points:result['status']='no_evaluation';continue
        complete=[]
        for s,v in points:
            requested=v.get('eval/requested_episodes',run.get('eval_episodes'))
            actual=v.get('eval/n_episodes',0)
            if requested is not None and actual==requested and actual>0:complete.append((s,v))
        if len(complete)!=len(points):warnings.append(str(path)+': incomplete or unknown evaluation counts excluded')
        if not complete:result['status']='incomplete_evaluation';continue
        terminal=next(((s,v) for s,v in complete if s>=budget),None)
        result['status']='complete' if terminal and (path/'DONE').exists() else 'unfinished'
        if terminal:complete=[(s,v) for s,v in complete if s<=terminal[0]]
        final=[v['eval/ep_rew_mean'] for s,v in complete if s>=budget*(1-tail)]
        result['last_step']=complete[-1][0]
        if terminal and final:result['final_score']=float(np.mean(final))
        x=np.array([s for s,v in complete]);y=np.array([v['eval/ep_rew_mean'] for s,v in complete])
        t=np.array([v.get('eval/session_wall_seconds',np.nan) for s,v in complete])
        # Resume clocks reset; never join sessions silently.
        time_ok=bool(np.isfinite(t).all() and np.all(np.diff(t)>0))
        offsets={s-v.get('eval/session_steps',s) for s,v in complete}
        time_ok=time_ok and len(offsets)==1
        result.update(step_auc=auc(x,y),step_auc_domain=[float(x[0]),float(x[-1])],
                      time_auc=auc(t,y) if time_ok else None,time_auc_domain=[float(t[0]),float(t[-1])] if time_ok else None)
        if not time_ok:warnings.append(str(path)+': missing or multi-session clock; time metrics unavailable')
        threshold=manifest.get('thresholds',{}).get(run['environment'])
        if threshold is not None:
            hits=np.flatnonzero(y>=threshold);result['threshold_reached']=bool(len(hits))
            result['threshold_steps']=float(x[hits[0]]) if len(hits) else None
            result['threshold_seconds']=float(t[hits[0]]) if len(hits) and time_ok else None
        result['source_hashes']={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ('config.json','environment.json','manifest.json') if (path/name).exists()}
        series[key]=(x,y,t if time_ok else None)
        for s,v in complete:
            score=v['eval/ep_rew_mean'];n=norm.get(run['environment'])
            curves.append(dict(algorithm=key[0],environment=key[1],seed=key[2],step=s,
                wall_seconds=v.get('eval/session_wall_seconds'),raw_score=score,
                normalized_score=(score-n['low'])/(n['high']-n['low']) if n else None))
        detail=path/'evaluation_episodes.jsonl'
        if detail.exists():
            means={float(v['step']):float(np.mean(v['returns'])) for v in map(json.loads,detail.read_text().splitlines()) if v.get('returns')}
            for s,v in complete:
                if s in means and not np.isclose(means[s],v['eval/ep_rew_mean'],rtol=1e-5,atol=1e-5):warnings.append(str(path)+f': episode mean mismatch at {s}')
    summaries=[];byalgo={}
    for algo,env in sorted({(r['algorithm'],r['environment']) for r in rows}):
        group=[r for r in rows if (r['algorithm'],r['environment'])==(algo,env)]
        if len({r['budget_steps'] for r in group})>1:raise ValueError('Mixed budgets in a seed group')
        vals=[r['final_score'] for r in group if r['status']=='complete' and r['final_score'] is not None]
        mean=float(np.mean(vals)) if vals else None;std=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        cv=std/abs(mean) if std is not None and abs(mean)>manifest.get('cv_mean_floor',1e-8) else None
        summaries.append(dict(algorithm=algo,environment=env,expected_seeds=len(group),complete_seeds=len(vals),
            failed_or_incomplete_seeds=len(group)-len(vals),mean=mean,std=std,cv=cv,mean_ci95=ci(vals,rng)))
        if env in norm and len(vals)==len(group):
            n=norm[env];byalgo.setdefault(algo,{})[env]=[(v-n['low'])/(n['high']-n['low']) for v in vals]
    aggregate={};required={r['environment'] for r in rows}
    for algo,data in byalgo.items():
        if set(data)!=required or len({len(v) for v in data.values()})!=1:
            warnings.append(algo+': IQM omitted for incomplete coverage or unequal seed counts');continue
        samples=[iqm(np.concatenate([rng.choice(v,len(v),replace=True) for v in data.values()])) for _ in range(2000)]
        aggregate[algo]=dict(iqm=iqm(np.concatenate(list(data.values()))),ci95=np.quantile(samples,[.025,.975]).tolist() if min(map(len,data.values()))>=2 else None)
    # Equal-wall-time checkpoint comparison on the COMMON observed interval.
    equal_time=[]
    for env in required:
        expected=[(r['algorithm'],r['environment'],r['seed']) for r in rows if r['environment']==env]
        if any(k not in series or series[k][2] is None for k in expected):continue
        left=max(series[k][2][0] for k in expected);right=min(series[k][2][-1] for k in expected)
        if right<left:continue
        for k in expected:
            x,y,t=series[k];grid=np.unique(np.r_[left,t[(t>left)&(t<right)],right]);vals=np.interp(grid,t,y)
            equal_time.append(dict(algorithm=k[0],environment=env,seed=k[2],domain=[float(left),float(right)],
                                   score_at_common_end=float(vals[-1]),auc=auc(grid,vals)))
    wins={};reference=manifest.get('comparison_reference');tol=float(manifest.get('tie_tolerance',0.))
    if reference:
        baseline={r['environment']:r for r in summaries if r['algorithm']==reference}
        for r in summaries:
            b=baseline.get(r['environment'])
            if r['algorithm']==reference or not b or r['mean'] is None or b['mean'] is None:continue
            if r['complete_seeds']!=r['expected_seeds'] or b['complete_seeds']!=b['expected_seeds']:continue
            diff=r['mean']-b['mean'];bucket='win' if diff>tol else 'loss' if diff < -tol else 'tie'
            wins.setdefault(r['algorithm'],dict(win=0,tie=0,loss=0))[bucket]+=1
    result=dict(runs=rows,summary=summaries,aggregate=aggregate,equal_time=equal_time,win_tie_loss=wins,warnings=warnings,
        conventions=dict(final_fraction=tail,normalization=norm,uncertainty='bootstrap training seeds within fixed tasks',
                         equal_time='linear interpolation inside common observed domain; no extrapolation',tie_tolerance=tol))
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    for name,data in [('curves.csv',curves),('summary.csv',summaries)]:
        if data:
            with (output/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('manifest');parser.add_argument('--output',required=True)
    args=parser.parse_args();build_report(json.loads(Path(args.manifest).read_text()),args.output)

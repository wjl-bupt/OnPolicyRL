"""Plot a stopped campaign without launching environments or training."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 state=json.loads((a.campaign/'status.json').read_text());a.output.mkdir(parents=True,exist_ok=True)
 rows=[];summary=[]
 for job in state['jobs']:
  path=Path(job['run_dir'])/'metrics.jsonl';seen={}
  if path.exists():
   for line in path.read_text().splitlines():
    try:r=json.loads(line)
    except json.JSONDecodeError:continue
    for metric,key in [('train','rollout'),('eval','eval')]:
     v=r.get(key,{}).get('ep_rew_mean')
     if v is not None and np.isfinite(v):seen[(metric,r['step'])]=dict(env=job['env'],algo=job['algo'],seed=job['seed'],status=job['status'],metric=metric,step=r['step'],return_mean=v,episodes=r.get(key,{}).get('n_episodes'))
  rows.extend(seen.values())
  summary.append({**{k:job[k] for k in ['env','algo','seed','status']},'last_step':max([x['step'] for x in seen.values()],default=0)})
 if not rows:raise ValueError('No recorded rewards')
 with (a.output/'curves.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (a.output/'status.json').write_text(json.dumps(summary,indent=2))
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 envs=sorted({r['env'] for r in rows});colors={'ppo':'#2864A0','ga2e':'#C76B21'}
 for metric in ['train','eval']:
  fig,axs=plt.subplots(2,3,figsize=(15,8));axs=axs.ravel()
  for ax,env in zip(axs,envs):
   for algo in ['ppo','ga2e']:
    rr=[r for r in rows if r['metric']==metric and r['env']==env and r['algo']==algo]
    seeds=sorted({r['seed'] for r in rr});curves={s:{r['step']:r['return_mean'] for r in rr if r['seed']==s} for s in seeds}
    complete=len(seeds)==5 and all(r['status']=='done' for r in rr)
    for s,c in curves.items():
     xx=sorted(c);ax.plot(np.array(xx)/1e6,[c[x] for x in xx],color=colors[algo],alpha=.18 if complete else .85,lw=.7 if complete else 1.3,ls=('-' if algo=='ppo' else '--') if complete else ['-','--',':','-.'][s%4],label=None if complete else f'{algo.upper()} seed {s} (partial)')
    if complete:
     common=sorted(set.intersection(*(set(c) for c in curves.values())))
     yy=np.array([[c[x] for x in common] for c in curves.values()]);mean=yy.mean(0);sd=yy.std(0,ddof=1);xx=np.array(common)/1e6
     ax.plot(xx,mean,color=colors[algo],lw=2,ls='-' if algo=='ppo' else '--',label=f'{algo.upper()} n=5')
     ax.fill_between(xx,mean-sd,mean+sd,color=colors[algo],alpha=.16)
   ax.set_title(env.replace('MinAtar/','').replace('-v0',''));ax.set_xlabel('Environment steps (millions)');ax.set_ylabel('Episode return');ax.set_xlim(0,10.1);ax.grid(alpha=.18);ax.legend(fontsize=8)
   if 'Breakout' in env and metric=='eval':ax.text(.02,.98,'Final evaluation: 40 episodes / seed',transform=ax.transAxes,va='top',fontsize=8)
  for ax in axs[len(envs):]:ax.axis('off')
  axs[-1].text(0,.9,'Recorded batch: JAX / Gymnax MinAtar\nPPO GAE lambda = 0.95\nNo added sticky-action wrapper\n\nComplete: mean +/- sample SD over 5 seeds\nFaint lines: individual seeds\nPartial: individual seeds only\nIndependent return scales across environments\nNo interpolation or smoothing\n\nTrain: logged rolling episode return\nEval: logged evaluation episode return',va='top',linespacing=1.6)
  fig.suptitle(f'MinAtar: {metric} curves | stopped batch',fontsize=17)
  fig.tight_layout(rect=(0,0,1,.95));fig.savefig(a.output/f'{metric}.png',dpi=180);fig.savefig(a.output/f'{metric}.pdf');plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

"""Real short PPO jobs from the public configurations, including eval/checkpoint."""
from pathlib import Path
import json
import sys
import torch
from workflow.experiment import expand, load_experiment, Job
from workflow.runner import run_job

path=Path(sys.argv[1])
job=expand(load_experiment(path))[0]
cfg=job.config
cfg['algo_params']=dict(num_envs=2,rollout_len=8,total_steps=32,num_mini_batch=2,
                        ppo_epoch=1,gae_lambda=.95,device='cpu',log_interval=1)
cfg['monitor']=dict(eval_num_envs=2,eval_episodes=2,eval_interval=0,ckpt_count=1)
cfg['env_params']['max_episode_steps']=4
out=Path('.integration-staging/backend_training')/path.stem
out.mkdir(parents=True,exist_ok=True)
job=Job('ppo',job.env,0,cfg,out,False,job.identity,job.label)
result=run_job(job,device='cpu')
if result['status']=='done':
    checkpoints=list((out/'model_pools').glob('*.pt'))
    assert checkpoints, 'checkpoint missing'
    state=torch.load(checkpoints[-1],weights_only=False,map_location='cpu')
    assert state['env_state'] is not None
    events=(out/'events.jsonl').read_text() if (out/'events.jsonl').exists() else ''
    assert 'eval_failed' not in events and 'checkpoint_failed' not in events, events
    rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines() if s]
    assert any(r.get('eval', {}).get('n_episodes')==2 for r in rows), 'two complete evaluation episodes required'
    result['normalizer_count']=state['env_state']['observation']['count']
(out/'validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
sys.exit(0 if result['status']=='done' else 1)

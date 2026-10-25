"""Bounded reset/step validation; records failures instead of hiding them."""
import json
import sys
from pathlib import Path
import torch
from workflow.experiment import load_experiment,expand
from common.env.factory import make_tensor_env

path=Path(sys.argv[1])
job=expand(load_experiment(path))[0]
params=dict(job.config['env_params'])
params['max_episode_steps']=3
env=None
try:
    env=make_tensor_env(job.env,num_envs=2,seed=7,device='cpu',**params)
    obs=env.reset(seed=7)
    assert tuple(obs.shape)==(2,*env.obs_space.shape), (obs.shape,env.obs_space)
    done_count=0
    for _ in range(4):
        a=torch.as_tensor(__import__('numpy').stack([env.action_space.sample() for _ in range(2)]))
        obs,r,m,info=env.step(a)
        assert torch.isfinite(obs).all() and torch.isfinite(r).all()
        assert info['_transition_obs'].shape==obs.shape
        done_count+=int((m.terminated|m.truncated).sum())
    assert done_count>=2
    result=dict(config=path.name,status='passed',shape=list(obs.shape),done_count=done_count,provenance=env.provenance)
except Exception as e:
    import traceback
    traceback.print_exc()
    result=dict(config=path.name,status='failed',error=repr(e))
finally:
    if env:env.close()
out=Path('.integration-staging/backend_validation');out.mkdir(exist_ok=True)
(out/(path.stem+'.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result))
sys.exit(0 if result['status']=='passed' else 1)

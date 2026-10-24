"""Exercise the real current framework from a disposable committed repository."""
from pathlib import Path
import json,shutil,subprocess,tempfile
from workflow.revision import run_revision

root=Path('/root/OnPolicyRL')
repo=Path(tempfile.mkdtemp(prefix='revision-smoke-',dir=root/'.integration-staging'))
shutil.copytree(root/'src',repo/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.STDOUT).decode().strip()
git('init','-q');git('config','user.name','OnPolicyRL validation');git('config','user.email','test@example.invalid')
git('add','src');git('commit','-qm','Validation-only source snapshot')
sha=git('rev-parse','HEAD')
cfg={'version':2,'family':'minatar','name':'family_revision_smoke',
 'algo':['ppo','ga2e'],'envs':['MinAtar/Breakout-v0'],'runs':1,
 'train':{'num_envs':4,'rollout_len':16,'total_steps':128,
 'num_mini_batch':2,'ppo_epoch':1,'device':'cpu','log_interval':1},
 'algorithms':{'ppo':{'train':{'gae_lambda':.95}},'ga2e':{'train':{'lambda_val':.99}}},
 'environment':{'backend':'jax','jax_platform':'cpu','sticky_action_prob':.1,'max_episode_steps':32},
 'evaluation':{'eval_num_envs':2,'eval_episodes':2,'eval_interval':0,'ckpt_count':1}}
config=repo/'local_config.json';config.write_text(json.dumps(cfg))
code=run_revision(config,sha,repo=repo,run_root=repo/'outputs')
manifest=json.loads(next((repo/'outputs').rglob('manifest.json')).read_text())
assert code==0 and manifest['status']=='done'
assert len(manifest['results']) == 2 and all(r['status'] == 'done' for r in manifest['results'])
assert manifest['executed_runner'].startswith(manifest['source_dir'])
assert (repo/'local_config.json').exists()
print(json.dumps({'commit':sha,'manifest':str(next((repo/'outputs').rglob('manifest.json'))),
 'status':manifest['status'],'results':manifest['results'],'note':'Disposable validation repository; main repository HEAD unchanged'},indent=2))
(root/'.integration-staging/revision_smoke_result.json').write_text(json.dumps(manifest,indent=2))

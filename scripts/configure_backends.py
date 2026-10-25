"""One-time migration: one environment-family file per available backend."""
from pathlib import Path
from copy import deepcopy
import time
import yaml

root = Path(__file__).resolve().parents[1]
directory = root/'configs'
backup = root/'.integration-staging'/f'configs_before_backends_{int(time.time())}'
backup.mkdir(parents=True)
old = {}
for p in directory.glob('*.yaml'):
    old[p.stem] = yaml.safe_load(p.read_text())
    (backup/p.name).write_bytes(p.read_bytes())

def write(family,backend,spec):
    spec = deepcopy(spec)
    spec.update(version=2,family=family)
    spec.pop('name',None); spec.pop('run_dir',None)
    spec.setdefault('environment',{}).update(backend=backend,autoreset_mode='SameStep')
    spec['environment'].setdefault('wrappers',{})
    if backend=='jax': spec['environment'].setdefault('jax_platform','cpu')
    else: spec['environment'].pop('jax_platform',None)
    (directory/f'{family}_{backend}.yaml').write_text('# Backend-specific experiment; algorithms and environments share this file.\n'+yaml.safe_dump(spec,sort_keys=False))

for backend in ['gymnasium','jax']:
    spec=deepcopy(old['minatar'])
    spec['environment']['implementation']='gymnasium' if backend=='gymnasium' else 'gymnax'
    write('minatar',backend,spec)
    spec=deepcopy(old['classic_control'])
    spec['environment']={'implementation':'gymnasium' if backend=='gymnasium' else 'gymnax','max_episode_steps':500,'wrappers':{'action_mode':'clip'}}
    write('classic_control',backend,spec)
    spec=deepcopy(old['mujoco'])
    spec['envs']=['InvertedPendulum-v5'] if backend=='gymnasium' else ['MJX/inverted_pendulum']
    spec['environment']['implementation']='gymnasium' if backend=='gymnasium' else 'brax-mjx'
    spec['environment']['wrappers']={'normalize_observation':True,'observation_clip':10.,'normalize_reward':False,'action_mode':'rescale'}
    write('mujoco',backend,spec)

template=dict(version=2,algo='ppo',runs=1,resume=False,
    train=dict(total_steps=1024,num_envs=2,rollout_len=32,num_mini_batch=2,ppo_epoch=2,gae_lambda=.95,device='cpu'),
    model=dict(net_arch=[64,64],lr=.0003),
    evaluation=dict(eval_interval=0,eval_episodes=2,eval_num_envs=2,ckpt_count=1))
for backend in ['gymnasium','jax']:
    spec=deepcopy(template)
    spec['envs']=['ALE/Pong-v5'] if backend=='gymnasium' else ['JAXAtari/pong']
    spec['environment']=dict(implementation='gymnasium' if backend=='gymnasium' else 'jaxtari',
        frame_skip=4,screen_size=84,grayscale=True,noop_max=30,terminal_on_life_loss=False,
        fire_reset=False,repeat_action_probability=.25,full_action_space=False,
        max_episode_steps=108000 if backend=='gymnasium' else 27000,
        wrappers=dict(frame_stack=4,reward_sign=False))
    if backend=='gymnasium': spec['environment']['frame_stack']=1
    write('atari',backend,spec)
    spec=deepcopy(template)
    spec['envs']=['DMControl/cartpole/balance'] if backend=='gymnasium' else ['Playground/CartpoleBalance']
    spec['environment']=dict(implementation='dm-control' if backend=='gymnasium' else 'playground-mjx',
        max_episode_steps=1000,action_repeat=1,wrappers=dict(action_mode='rescale'))
    if backend=='gymnasium': spec['environment']['obs_mode']='state'
    write('dmcontrol',backend,spec)
    spec=deepcopy(template)
    spec['envs']=['MiniGrid-Empty-5x5-v0'] if backend=='gymnasium' else ['XMiniGrid/MiniGrid-Empty-5x5']
    spec['environment']=dict(implementation='gymnasium' if backend=='gymnasium' else 'xminigrid',max_episode_steps=100)
    if backend=='gymnasium': spec['environment']['obs_mode']='symbolic'
    write('minigrid',backend,spec)
    spec=deepcopy(template)
    spec['envs']=['Craftax-Classic-Symbolic-v1']
    spec['environment']=dict(implementation='craftax',max_episode_steps=1000)
    write('craftax',backend,spec)

for family,envs in [('box2d',['LunarLander-v3']),('toy_text',['FrozenLake-v1','Taxi-v3'])]:
    spec=deepcopy(template);spec['envs']=envs
    spec['environment']=dict(implementation='gymnasium',max_episode_steps=500)
    write(family,'gymnasium',spec)
for name in ['minatar','mujoco','classic_control']:
    (directory/f'{name}.yaml').unlink()
print('Original configs preserved:',backup)

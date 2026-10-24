"""One-time, lossless migration of the historical configuration collection."""
from pathlib import Path
import shutil
import time
import yaml

root = Path(__file__).resolve().parents[1]
configs = root / 'configs'
backup = root / '.integration-staging' / f'configs_before_family_{int(time.time())}'
backup.mkdir(parents=True)
old = {p.relative_to(configs): p.read_bytes() for p in configs.rglob('*.yaml')}


def convert(path, family):
    data = yaml.safe_load((configs / path).read_text())
    base = data.get('baseline', {})
    result = dict(version=2, family=family, algo=data['algo'], envs=data['envs'],
                  runs=data.get('runs', 1), resume=data.get('resume', False))
    names = dict(algo_params='train', network_params='model', env_params='environment', monitor='evaluation')
    result.update({names[k]: v for k, v in base.items()})
    if data.get('space'):
        result['sweep'] = {names[k.split('.')[0]] + '.' + k.split('.', 1)[1]: v for k, v in data['space'].items()}
    return result


minatar = convert('jax_minatar_compare/ppo_Asterix.yaml', 'minatar')
minatar['algo'] = ['ppo', 'ga2e']
minatar['envs'] = ['MinAtar/' + e + '-v0' for e in ['Asterix', 'Breakout', 'Freeway', 'Seaquest', 'SpaceInvaders']]
minatar['train'].pop('gae_lambda')
minatar['algorithms'] = {'ppo': {'train': {'gae_lambda': .95}},
                         'ga2e': {'train': {'lambda_val': .99, 'lambda_refresh': 'epoch', 'grad_chunk_size': 2048}}}
minatar['environments'] = {'MinAtar/Freeway-v0': {'environment': {'max_episode_steps': 2500}}}
classic = convert('cartpole_ppo.yaml', 'classic_control')
classic['train']['gae_lambda'] = .95
mujoco = convert('mjx_inverted_pendulum_ppo.yaml', 'mujoco')
for rel, content in old.items():
    dest = backup / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
for rel in old:
    (configs / rel).unlink()
for name, spec in [('minatar', minatar), ('classic_control', classic), ('mujoco', mujoco)]:
    (configs / f'{name}.yaml').write_text('# One environment family; select algorithms and environments here.\n' + yaml.safe_dump(spec, sort_keys=False))
for directory in sorted(configs.rglob('*'), reverse=True):
    if directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
print(f'Preserved {len(old)} original configurations in {backup}')

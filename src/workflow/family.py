"""Compile a family declaration into the same jobs used by training and reporting.

No YAML inheritance or algorithm-specific configuration files are required.
Merge order: shared sections -> algorithm delta -> environment delta -> sweep.
"""
from copy import deepcopy
from dataclasses import fields
from itertools import product
from pathlib import Path

SECTIONS = {'train': 'algo_params', 'model': 'network_params',
            'environment': 'env_params', 'evaluation': 'monitor'}
TOP = {'version', 'family', 'name', 'algo', 'envs', 'runs', 'resume', 'run_dir',
       'algorithms', 'environments', 'sweep', *SECTIONS}


def mapping(value, where):
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be a mapping')
    return value


def merge(base, delta):
    result = deepcopy(base)
    for key, value in delta.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else deepcopy(value)
    return result


def sections(value, where):
    value = mapping(value, where)
    if set(value) - SECTIONS.keys():
        raise ValueError(f'{where}: unknown sections {sorted(set(value) - SECTIONS.keys())}')
    return {SECTIONS[k]: deepcopy(mapping(v, f'{where}.{k}')) for k, v in value.items()}


def parse_family(data, path):
    from workflow.experiment import Experiment
    if set(data) - TOP:
        raise ValueError(f'unknown family configuration keys: {sorted(set(data) - TOP)}')
    if data.get('version') != 2:
        raise ValueError('family configuration requires version: 2')
    if data.get('family') not in ('minatar', 'mujoco', 'classic_control'):
        raise ValueError('family must be minatar, mujoco or classic_control')
    algos = data.get('algo')
    algos = [algos] if isinstance(algos, str) else algos
    envs = data.get('envs')
    for label, values in [('algo', algos), ('envs', envs)]:
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f'{label} must be a non-empty string list (algo also accepts a string)')
        if len(values) != len(set(values)):
            raise ValueError(f'{label} contains duplicates')
    runs = data.get('runs', 1)
    if type(runs) is not int or runs < 1:
        raise ValueError('runs must be a positive integer')
    if type(data.get('resume', False)) is not bool:
        raise ValueError('resume must be a boolean')
    spec = deepcopy(data)
    spec['algo'] = algos
    name = data.get('name', Path(path).stem)
    return Experiment(','.join(algos), envs, runs, data.get('resume', False),
                      Path(data.get('run_dir', f'runs/{name}')), {}, {}, name, spec)


def compile_family(exp):
    from workflow.experiment import Job, _identity, _env_slug, _set_by_path
    from workflow.registry import get_algo, list_algorithms
    from architecture.actor_critic import PolicyConfig
    from common.env.minatar_options import sticky_probability
    import inspect
    from workflow.monitor import Monitor
    spec = exp.family_spec
    shared = sections({k: spec[k] for k in SECTIONS if k in spec}, 'shared')
    algorithms = mapping(spec.get('algorithms', {}), 'algorithms')
    environments = mapping(spec.get('environments', {}), 'environments')
    for name, delta in algorithms.items():
        if name not in list_algorithms():
            raise ValueError(f'unknown algorithm override: {name}')
        sections(delta, f'algorithms.{name}')
    for name, delta in environments.items():
        sections(delta, f'environments.{name}')
    sweep = mapping(spec.get('sweep', {}), 'sweep')
    for key, values in sweep.items():
        head, sep, tail = key.partition('.')
        if head not in SECTIONS or not sep or not tail or not isinstance(values, list) or not values:
            raise ValueError(f'invalid sweep {key!r}: use section.field: [candidates]')
    jobs = []
    for algo in spec['algo']:
        entry = get_algo(algo)
        base = merge(shared, sections(algorithms.get(algo, {}), f'algorithms.{algo}'))
        for env in exp.envs:
            if spec['family'] == 'minatar' and not env.startswith('MinAtar/'):
                raise ValueError(f'{env}: minatar requires MinAtar/... environment IDs')
            if spec['family'] != 'minatar' and env.startswith('MinAtar/'):
                raise ValueError(f'{env}: wrong environment family')
            for values in product(*sweep.values()):
                config = merge(base, sections(environments.get(env, {}), f'environments.{env}'))
                for key, value in zip(sweep, values):
                    head, tail = key.split('.', 1)
                    _set_by_path(config, SECTIONS[head] + '.' + tail, value)
                params = config.setdefault('env_params', {})
                if spec['family'] == 'minatar':
                    params['sticky_action_prob'] = sticky_probability(params.get('sticky_action_prob', .1))
                elif 'sticky_action_prob' in params:
                    raise ValueError('sticky_action_prob is only valid for minatar')
                for section, valid in [('algo_params', {f.name for f in fields(entry.config_cls)} - {'seed'}),
                                       ('network_params', {f.name for f in fields(PolicyConfig)}),
                                       ('monitor', (set(inspect.signature(Monitor).parameters) - {'policy', 'env_factory', 'log', 'run_dir', 'total_steps', 'meta', 'eval_fn', 'save_fn'}) | {'eval_num_envs'})]:
                    unknown = set(config.get(section, {})) - valid
                    if unknown:
                        raise ValueError(f'{algo}/{env}/{section}: unknown parameters {sorted(unknown)}')
                # One group per algorithm/configuration; seeds share a group hash.
                group_config = {'base': base, 'environments': environments, 'sweep': dict(zip(sweep, values))}
                group = algo + '-' + _identity(algo, '', 0, group_config)
                for seed in range(exp.runs):
                    identity = _identity(algo, env, seed, config)
                    jobs.append(Job(algo, env, seed, deepcopy(config),
                                    exp.run_dir / group / _env_slug(env) / f'seed-{seed}',
                                    exp.resume, identity, f'{group}|{env}|seed-{seed}'))
    identities = [j.identity for j in jobs]
    if len(identities) != len(set(identities)):
        raise ValueError('sweep generates duplicate jobs')
    return jobs

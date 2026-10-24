import copy
import json
import pytest
from workflow.experiment import load_experiment, expand


def plan(tmp_path, data):
    p = tmp_path / 'family.json'
    p.write_text(json.dumps(data))
    return expand(load_experiment(p))


@pytest.fixture
def spec():
    return dict(version=2, family='minatar', algo=['ppo', 'ga2e'],
                envs=['MinAtar/Asterix-v0', 'MinAtar/Freeway-v0'], runs=5,
                train={'total_steps': 10000000},
                environment={'backend': 'jax'},
                algorithms={'ppo': {'train': {'gae_lambda': .95}},
                            'ga2e': {'train': {'lambda_val': .99}}},
                environments={'MinAtar/Freeway-v0': {'environment': {'max_episode_steps': 2500}}})


def test_matrix_and_isolation(tmp_path, spec):
    jobs = plan(tmp_path, spec)
    assert len(jobs) == 20
    assert len({j.run_dir for j in jobs}) == 20
    assert all(j.config['env_params']['sticky_action_prob'] == .1 for j in jobs)
    for j in jobs:
        p = j.config['algo_params']
        assert ('gae_lambda' in p) == (j.algo == 'ppo')
        assert ('lambda_val' in p) == (j.algo == 'ga2e')
        assert ('max_episode_steps' in j.config['env_params']) == ('Freeway' in j.env)
    jobs[0].config['algo_params']['total_steps'] = 1
    assert jobs[1].config['algo_params']['total_steps'] == 10000000


def test_sweep_precedence_and_identity(tmp_path, spec):
    old = plan(tmp_path, spec)
    spec['sweep'] = {'train.total_steps': [128, 256]}
    new = plan(tmp_path, spec)
    assert len(new) == 40
    assert {j.config['algo_params']['total_steps'] for j in new} == {128, 256}
    assert not {j.run_dir for j in old} & {j.run_dir for j in new}


@pytest.mark.parametrize('mutation', [
    {'train': {'typo': 2}}, {'algo': ['ppo', 'ppo']},
    {'sweep': {'train.total_steps': []}}, {'runs': 1.5},
    {'environment': {'sticky_action_prob': 1.1}},
    {'algorithms': {'unknown': {}}}, {'version': 3},
    {'sweep': {'train.total_steps': [1, 1]}},
])
def test_invalid_fails_before_training(tmp_path, spec, mutation):
    spec.update(mutation)
    with pytest.raises(ValueError):
        plan(tmp_path, spec)


def test_non_minatar_has_no_sticky(tmp_path):
    data = dict(version=2, family='classic_control', algo='ppo', envs=['CartPole-v1'])
    assert 'sticky_action_prob' not in plan(tmp_path, data)[0].config['env_params']
    data['environment'] = {'sticky_action_prob': .1}
    with pytest.raises(ValueError):
        plan(tmp_path, data)

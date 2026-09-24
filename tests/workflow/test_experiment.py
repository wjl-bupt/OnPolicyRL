"""Experiment loading + grid expansion -- workflow/experiment.py (L4).

expand() is the "排列组合": the Cartesian product of `space`, deep-merged onto
`baseline`, then fanned across every env x run seed. These tests pin the product
size and ORDER (experiment -> env -> seed), the fixed 0..runs-1 seed sequence,
the baseline/space separation, and the fail-fast validation.
"""

from pathlib import Path

import pytest

from workflow.experiment import expand, load_experiment

GRID = """
algo: ppo
envs: [CartPole-v1]
runs: 2
resume: false
run_dir: runs/x
baseline:
  algo_params: {rollout_len: 128, total_steps: 1000}
  network_params: {net_arch: [64, 64], lr: 0.0003}
space:
  algo_params.epsilon: [0.1, 0.2, 0.3]
  network_params.lr: [0.0003, 0.001]
"""


def test_cartesian_product_size_and_overrides(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text(GRID)
    exp = load_experiment(p)
    jobs = expand(exp)
    assert len(jobs) == 3 * 2 * 1 * 2        # epsilon x lr x env x runs

    j = jobs[0]
    assert j.config["algo_params"]["epsilon"] in (0.1, 0.2, 0.3)
    assert j.config["network_params"]["lr"] in (0.0003, 0.001)
    # a list in baseline stays a fixed value; baseline scalars untouched
    assert j.config["network_params"]["net_arch"] == [64, 64]
    assert j.config["algo_params"]["rollout_len"] == 128

    # both epsilon values actually appear across the grid (the sweep happened)
    seen = {job.config["algo_params"]["epsilon"] for job in jobs}
    assert seen == {0.1, 0.2, 0.3}

    # identities and run dirs are unique per (combo, env, seed)
    assert len({job.identity for job in jobs}) == len(jobs)
    assert len({str(job.run_dir) for job in jobs}) == len(jobs)


def test_runs_pins_seeds_to_zero_based_sequence(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("algo: ppo\nenvs: [X]\nruns: 3\n"
                 "baseline: {algo_params: {total_steps: 1}}\n")
    jobs = expand(load_experiment(p))
    assert [j.seed for j in jobs] == [0, 1, 2]      # fixed 0..runs-1, not hand-picked


def test_envs_multiply_and_emission_order(tmp_path):
    # 2 experiments (epsilon) x 2 envs x 2 runs = 8, ordered experiment->env->seed
    p = tmp_path / "e.yaml"
    p.write_text("algo: ppo\nenvs: [A, B]\nruns: 2\nrun_dir: runs/o\n"
                 "baseline: {algo_params: {total_steps: 1}}\n"
                 "space: {algo_params.epsilon: [0.1, 0.2]}\n")
    jobs = expand(load_experiment(p))
    assert len(jobs) == 2 * 2 * 2
    order = [(j.config["algo_params"]["epsilon"], j.env, j.seed) for j in jobs]
    assert order == [
        (0.1, "A", 0), (0.1, "A", 1), (0.1, "B", 0), (0.1, "B", 1),
        (0.2, "A", 0), (0.2, "A", 1), (0.2, "B", 0), (0.2, "B", 1),
    ]


def test_env_id_with_slash_is_filesystem_safe(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("algo: ppo\nenvs: ['MinAtar/Breakout-v0']\nrun_dir: runs/m\n"
                 "baseline: {algo_params: {total_steps: 1}}\n")
    j = expand(load_experiment(p))[0]
    assert j.env == "MinAtar/Breakout-v0"           # the id is preserved on the job
    assert "/" not in j.run_dir.name                # ...but the dir name is safe
    assert "MinAtar-Breakout-v0" in str(j.run_dir)


def test_env_scalar_sugar_and_runs_default(tmp_path):
    # `env` (scalar) is 1-element sugar for `envs`; omitted `runs` defaults to 1
    p = tmp_path / "e.yaml"
    p.write_text("algo: ppo\nenv: CartPole-v1\n"
                 "baseline: {algo_params: {total_steps: 1}}\n")
    exp = load_experiment(p)
    assert exp.envs == ["CartPole-v1"] and exp.runs == 1
    jobs = expand(exp)
    assert len(jobs) == 1 and jobs[0].seed == 0
    assert "base" in str(jobs[0].run_dir)


def test_scalar_space_value_is_rejected(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("algo: ppo\nenvs: [X]\nspace: {algo_params.epsilon: 0.2}\n")
    with pytest.raises(ValueError, match="list of candidate"):
        expand(load_experiment(p))


def test_unknown_top_level_key_is_rejected(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("algo: ppo\nenvs: [X]\nbogus_key: 1\n")
    with pytest.raises(ValueError, match="unknown top-level"):
        load_experiment(p)


def test_missing_algo_or_env_is_rejected(tmp_path):
    p = tmp_path / "e.yaml"                          # no algo
    p.write_text("envs: [X]\n")
    with pytest.raises(ValueError, match="algo"):
        load_experiment(p)
    p2 = tmp_path / "e2.yaml"                        # no envs / env
    p2.write_text("algo: ppo\n")
    with pytest.raises(ValueError, match="envs"):
        load_experiment(p2)


def test_run_dir_defaults_to_runs_slash_name(tmp_path):
    # no run_dir declared -> derived as runs/<name>; name defaults to the file stem
    p = tmp_path / "my_exp.yaml"
    p.write_text("algo: ppo\nenvs: [X]\nbaseline: {algo_params: {total_steps: 1}}\n")
    assert load_experiment(p).run_dir == Path("runs") / "my_exp"

    # an explicit `name` wins over the file stem
    p2 = tmp_path / "e.yaml"
    p2.write_text("algo: ppo\nname: sweepA\nenvs: [X]\n"
                  "baseline: {algo_params: {total_steps: 1}}\n")
    assert load_experiment(p2).run_dir == Path("runs") / "sweepA"


def test_explicit_run_dir_overrides(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("algo: ppo\nenvs: [X]\nrun_dir: custom/place\n"
                 "baseline: {algo_params: {total_steps: 1}}\n")
    assert load_experiment(p).run_dir == Path("custom/place")

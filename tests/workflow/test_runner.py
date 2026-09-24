"""Runner end-to-end + resume toggle -- workflow/runner.py (L4).

A tiny CartPole grid runs to completion; then a re-run with resume=true skips the
finished job. This is the whole vertical slice exercised through one entry point.
"""

from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

from workflow.runner import run

# num_envs*rollout_len=32 per update; ~8 updates; eval/ckpt only on the final
# iteration (intervals well above the update count) -> a few seconds on CPU.
CFG = """
algo: ppo
envs: [CartPole-v1]
runs: 1
resume: true
run_dir: {run_dir}
baseline:
  algo_params:
    num_envs: 2
    rollout_len: 16
    ppo_epoch: 1
    num_mini_batch: 2
    total_steps: 256
    log_interval: 1
  network_params:
    net_arch: [32]
    lr: 0.001
  monitor:
    eval_interval: 100
    eval_episodes: 1
    ckpt_interval: 100
space:
  algo_params.epsilon: [0.2]
"""


def test_runner_end_to_end_then_resume_skips(tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(CFG.format(run_dir=str(tmp_path / "runs")))

    summaries = run(cfg)
    assert len(summaries) == 1
    assert summaries[0]["status"] == "done", summaries[0]

    # [1] the launched config is snapshotted verbatim under the run root
    assert (tmp_path / "runs" / "experiment.snapshot.yaml").exists()

    job_dir = Path(summaries[0]["run_dir"])
    for f in ("config.json", "metrics.jsonl", "run.log", "DONE"):
        assert (job_dir / f).exists(), f"missing artifact: {f}"
    # [3][4b] the checkpoint lives in the model pool, named by global_step
    pool = list((job_dir / "model_pools").glob("model_step-*.pt"))
    assert pool, "no model_step-*.pt in the model pool"

    # resume=true + a DONE marker -> the job is skipped on a re-run
    again = run(cfg)
    assert again[0]["status"] == "skipped"


def test_runner_resume_loads_from_model_pool(tmp_path):
    # a finished run leaves a model pool; dropping DONE forces the re-run to load
    # the newest checkpoint (fingerprint matches) instead of skipping.
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(CFG.format(run_dir=str(tmp_path / "runs")))

    s1 = run(cfg)
    job_dir = Path(s1[0]["run_dir"])
    assert list((job_dir / "model_pools").glob("model_step-*.pt"))
    (job_dir / "DONE").unlink()                 # no longer "finished" -> must resume

    s2 = run(cfg)                               # loads latest ckpt, runs to completion
    assert s2[0]["status"] == "done", s2[0]

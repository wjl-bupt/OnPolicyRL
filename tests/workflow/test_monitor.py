"""Monitor orchestration -- workflow/monitor.py (L4).

The eval / checkpoint primitives are injected as stubs so the orchestration
(anomaly reporting, interval firing, final-iteration forcing, and the no-raise
guarantee) is tested without a live env.
"""

from pathlib import Path

from workflow.monitor import Monitor
from common.logger import Logger


class _StubPolicy:
    """Monitor only forwards it to the injected eval_fn / save_fn stubs."""


def _monitor(tmp_path, log, **kw):
    defaults = dict(total_steps=1000, eval_interval=0, ckpt_interval=0,
                    eval_fn=lambda *a, **k: {}, save_fn=lambda *a, **k: None)
    defaults.update(kw)
    return Monitor(_StubPolicy(), env_factory=lambda: None, log=log,
                   run_dir=tmp_path, **defaults)


def test_anomalies_are_reported_never_raised(tmp_path):
    log = Logger(run_dir=tmp_path)
    mon = _monitor(tmp_path, log)
    # NaN metric + KL explosion; observe must return None, not raise
    assert mon.observe({"train/kl": 5.0, "loss/policy": float("nan")},
                       global_step=100, iteration=1) is None
    log.close()
    anomalies = (tmp_path / "anomalies.jsonl").read_text(encoding="utf-8")
    assert "kl_explode" in anomalies
    assert "nonfinite_metric" in anomalies


def test_eval_and_ckpt_fire_on_their_intervals(tmp_path):
    log = Logger(run_dir=tmp_path)
    evals, ckpts = [], []
    mon = _monitor(tmp_path, log, eval_interval=5, ckpt_interval=10,
                   eval_fn=lambda *a, **k: evals.append(1) or {"eval/ep_rew_mean": 1.0},
                   save_fn=lambda *a, **k: ckpts.append(1))
    mon.observe({}, global_step=100, iteration=5)     # eval only
    mon.observe({}, global_step=200, iteration=7)     # neither
    mon.observe({}, global_step=300, iteration=10)    # eval + ckpt
    log.close()
    assert len(evals) == 2 and len(ckpts) == 1


def test_final_iteration_forces_eval_and_ckpt(tmp_path):
    log = Logger(run_dir=tmp_path)
    evals, ckpts = [], []
    mon = _monitor(tmp_path, log, eval_interval=999, ckpt_interval=999,
                   eval_fn=lambda *a, **k: evals.append(1) or {},
                   save_fn=lambda *a, **k: ckpts.append(1))
    # step >= total_steps -> final, regardless of the (huge) intervals
    mon.observe({}, global_step=1000, iteration=3)
    log.close()
    assert len(evals) == 1 and len(ckpts) == 1


def test_eval_failure_does_not_raise(tmp_path):
    log = Logger(run_dir=tmp_path)

    def _boom(*a, **k):
        raise RuntimeError("eval blew up")

    mon = _monitor(tmp_path, log, eval_interval=1, eval_fn=_boom)
    assert mon.observe({}, global_step=10, iteration=1) is None    # swallowed
    log.close()
    assert "eval_failed" in (tmp_path / "anomalies.jsonl").read_text(encoding="utf-8")


def test_checkpoint_goes_to_model_pool_named_by_step(tmp_path):
    # the path monitor hands to save_fn: run_dir/model_pools/model_step-<gs>.pt
    log = Logger(run_dir=tmp_path)
    saved = {}
    mon = _monitor(tmp_path, log, ckpt_interval=1,
                   save_fn=lambda path, *a, **k: saved.setdefault("path", Path(path)))
    mon.observe({}, global_step=2048, iteration=1)     # final -> ckpt fires
    log.close()
    assert saved["path"].parent.name == "model_pools"
    assert saved["path"].name == "model_step-2048.pt"

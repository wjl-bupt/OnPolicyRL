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
    # ckpt_count=0 by default -> legacy periodic ckpt_interval mode, so the
    # interval-focused tests below are not pre-empted by the count schedule.
    defaults = dict(total_steps=1000, eval_interval=0, ckpt_count=0, ckpt_interval=0,
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


# ------------------------- count-mode checkpointing ---------------------- #

def _collect_ckpts(tmp_path, **kw):
    """Run 20 updates (global_step +50 each, total_steps=1000) and return the
    list of global_steps handed to save_fn -- i.e. the checkpoint schedule."""
    run_dir = Path(tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    saved: list[int] = []
    log = Logger(run_dir=run_dir)
    mon = _monitor(run_dir, log, save_fn=lambda path, *a, **k: saved.append(
        int(Path(path).stem.split("-")[1])), **kw)
    gs = 0
    for it in range(1, 21):
        gs += 50
        mon.observe({}, global_step=gs, iteration=it)
    log.close()
    return saved


def test_checkpoint_count_saves_first_middle_final(tmp_path):
    # default 3 saves: first rollout / one middle / final -- keeps the pool tiny
    assert _collect_ckpts(tmp_path, ckpt_count=3) == [50, 500, 1000]


def test_checkpoint_count_is_adjustable(tmp_path):
    # count is a knob: 1 -> final only, 2 -> first+final, 5 -> first+3 middles+final
    assert _collect_ckpts(tmp_path / "c1", ckpt_count=1) == [1000]
    assert _collect_ckpts(tmp_path / "c2", ckpt_count=2) == [50, 1000]
    assert _collect_ckpts(tmp_path / "c5", ckpt_count=5) == [50, 250, 500, 750, 1000]


def test_checkpoint_count_zero_falls_back_to_interval(tmp_path):
    # count<=0 restores the legacy periodic mode (final is still always saved)
    assert _collect_ckpts(tmp_path, ckpt_count=0, ckpt_interval=3) == \
        [150, 300, 450, 600, 750, 900, 1000]


def test_checkpoint_count_dedups_repeated_step(tmp_path):
    # the same global_step is never written twice (defensive dedup)
    run_dir = tmp_path
    log = Logger(run_dir=run_dir)
    saved = []
    mon = _monitor(run_dir, log, ckpt_count=3,
                   save_fn=lambda path, *a, **k: saved.append(int(
                       Path(path).stem.split("-")[1])))
    mon.observe({}, global_step=1000, iteration=1)     # first AND final at once
    mon.observe({}, global_step=1000, iteration=2)     # same step -> no 2nd write
    log.close()
    assert saved == [1000]

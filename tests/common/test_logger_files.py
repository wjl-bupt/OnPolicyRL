"""The three human-readable text logs -- common/logger.py (L1).

Beside the two machine files (metrics.jsonl / anomalies.jsonl), a run_dir logger
keeps run.log (messages + metric tables), anomaly.log (readable events), and
error.log (crash tracebacks). None of them may replace the jsonl files.
"""

from common.logger import Logger


def test_run_anomaly_and_error_logs(tmp_path):
    log = Logger(run_dir=tmp_path)
    log.info("hello world", step=1)
    log.record("loss/policy", 0.5)
    log.record("rollout/ep_rew_mean", 42.0)
    log.dump(10)
    log.event("kl_explode", iteration=1, global_step=10, value=0.9, threshold=0.5)
    log.crash("Traceback (most recent call last):\n  ... boom")
    log.close()

    run_log = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "hello world" in run_log                 # messages persisted
    assert "ep_rew_mean" in run_log                 # metric tables persisted

    anomaly_log = (tmp_path / "anomaly.log").read_text(encoding="utf-8")
    assert "kl_explode" in anomaly_log

    error_log = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert "boom" in error_log

    # the machine files still exist and are unaffected
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "anomalies.jsonl").exists()


def test_no_run_dir_means_no_files(tmp_path):
    """Console-only logger (run_dir=None) writes no files -- unchanged behavior."""
    log = Logger()
    log.info("just console")
    log.crash("nowhere to write")
    log.close()
    assert not list(tmp_path.iterdir())

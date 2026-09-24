"""Result aggregation -- common/results.py (L1).

Synthetic metrics.jsonl files pin the metric definitions (tail-window means,
last-recorded eval), the step-dedup, and the cross-seed mean/std reduction.
"""

import json
import math

from common.results import group_result, seed_result


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_seed_result_metric_definitions(tmp_path):
    # a 10-point rollout curve 1..10; eval recorded only at steps 5 and 10
    rows = []
    for i in range(1, 11):
        row = {"step": i * 100, "rollout/ep_rew_mean": float(i)}
        if i in (5, 10):
            row["eval/ep_rew_mean"] = float(i * 10)     # 50 @500, 100 @1000
        rows.append(row)
    # a trailing flush line repeating the last step with ONLY rollout (dup step)
    rows.append({"step": 1000, "rollout/ep_rew_mean": 10.0})
    p = tmp_path / "metrics.jsonl"
    _write_jsonl(p, rows)

    sr = seed_result(0, p)
    assert sr.n_points == 10                    # dup step collapses -> 10 unique
    assert sr.rew_last10 == 10.0                # ceil(10*0.1)=1 -> [10]
    assert sr.rew_last30 == 9.0                 # ceil(10*0.3)=3 -> mean(8,9,10)
    assert sr.final_eval == 100.0               # last line WITH eval, not the flush


def test_missing_file_is_all_nan(tmp_path):
    sr = seed_result(3, tmp_path / "nope.jsonl")
    assert sr.n_points == 0
    assert math.isnan(sr.rew_last10) and math.isnan(sr.final_eval)


def test_group_cross_seed_mean_std(tmp_path):
    def mk(seed, ev):
        p = tmp_path / f"m{seed}.jsonl"
        _write_jsonl(p, [{"step": 100, "rollout/ep_rew_mean": 5.0,
                          "eval/ep_rew_mean": ev}])
        return seed_result(seed, p)

    g = group_result("base", "X", [mk(0, 100.0), mk(1, 200.0)])
    assert g.final_eval.mean == 150.0
    assert g.final_eval.std == 50.0             # population std of {100, 200}
    assert g.final_eval.n == 2


def test_nan_seed_excluded_from_stats(tmp_path):
    p0 = tmp_path / "m0.jsonl"
    _write_jsonl(p0, [{"step": 100, "rollout/ep_rew_mean": 5.0, "eval/ep_rew_mean": 42.0}])
    p1 = tmp_path / "m1.jsonl"
    _write_jsonl(p1, [{"step": 100, "rollout/ep_rew_mean": 5.0}])    # no eval
    g = group_result("base", "X", [seed_result(0, p0), seed_result(1, p1)])
    assert g.final_eval.mean == 42.0 and g.final_eval.n == 1         # NaN seed dropped

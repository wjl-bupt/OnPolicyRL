"""Results report -- workflow/report.py (L4).

A synthetic run tree (one metrics.jsonl per seed dir, at the paths expand()
computes) is aggregated into the three-piece. Pins results.json structure +
ranking order, summary.md content, and the matplotlib guard (a missing install
skips the figure instead of raising).
"""

import builtins
import json

from common.logger import Logger
from common.results import group_result, seed_result
from workflow.experiment import load_experiment
from workflow.report import plot_curves, report

CFG = """
algo: ppo
envs: [EnvA]
runs: 2
run_dir: __RUN_DIR__
baseline:
  algo_params: {total_steps: 1}
space:
  algo_params.epsilon: [0.1, 0.2]
"""


def _write_metrics(d, ev):
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"step": 100, "rollout/ep_rew_mean": 10.0},
            {"step": 200, "rollout/ep_rew_mean": 20.0, "eval/ep_rew_mean": ev}]
    (d / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_report_three_piece_and_ranking(tmp_path):
    run_dir = tmp_path / "runs"
    p = tmp_path / "exp.yaml"
    p.write_text(CFG.replace("__RUN_DIR__", str(run_dir)))
    exp = load_experiment(p)

    # synth metrics at the exact expand() paths: eps=0.2 clearly beats eps=0.1
    for slug, seedvals in {"epsilon=0.1": [100.0, 120.0],
                           "epsilon=0.2": [400.0, 420.0]}.items():
        for seed, ev in enumerate(seedvals):
            _write_metrics(run_dir / slug / "EnvA" / f"seed-{seed}", ev)

    report(exp)

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["runs"] == 2 and results["envs"] == ["EnvA"]
    assert len(results["groups"]) == 2

    ranking = results["ranking"]                       # sorted by final_eval mean
    assert ranking[0]["experiment"] == "epsilon=0.2" and ranking[0]["rank"] == 1
    assert ranking[0]["final_eval_mean"] == 410.0
    assert ranking[1]["experiment"] == "epsilon=0.1"

    md = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "## EnvA" in md and "epsilon=0.2" in md

    # matplotlib is installed here -> a per-env curve PNG is produced
    assert (run_dir / "curve_EnvA.png").exists()


def test_plot_guard_when_matplotlib_missing(tmp_path, monkeypatch):
    # a real group so plotting WOULD be attempted if the guard were absent
    d = tmp_path / "s0"
    d.mkdir()
    (d / "m.jsonl").write_text(
        json.dumps({"step": 1, "rollout/ep_rew_mean": 5.0}) + "\n", encoding="utf-8")
    g = group_result("base", "EnvA", [seed_result(0, d / "m.jsonl")])

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("simulated missing matplotlib")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = plot_curves([g], tmp_path, log=Logger())     # must not raise
    assert out == []

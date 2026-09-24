# -*- encoding: utf-8 -*-
"""Stage (4) report: reduce a finished sweep into the results three-piece.

    results.json  -- machine-readable: per (experiment, env) stats + a flat ranking
    summary.md    -- human ranking table, one section per env
    curve_<env>.png -- one figure per env, a line per experiment (cross-seed
                       mean +/- std band)

It reads ONLY on-disk artifacts (each job's ``metrics.jsonl``) and never touches
L2 -- the experiment declaration drives which run dirs to read (via ``expand``),
so a stale directory left by an earlier layout is simply not looked at. The plot
is an OPTIONAL capability: matplotlib is imported lazily and a missing install
downgrades to "skip the figure, warn", never an error (report-only health).

Aggregation unit = one (experiment slug, env) across its ``runs`` seeds, which is
exactly the ``run_dir/<slug>/<env>/seed-N`` layout expand() produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python src/workflow/report.py experiment.yaml`: put `src` on the path
# BEFORE the framework imports below resolve. Under pytest, conftest already did.
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import math
import statistics
from collections import defaultdict

from common.logger import Logger
from common.results import GroupResult, group_result, seed_result
from workflow.experiment import Experiment, _env_slug, expand, load_experiment


# --------------------------------------------------------------------------- #
#  grouping jobs into (experiment, env) buckets
# --------------------------------------------------------------------------- #


def _parse_slug(slug: str) -> dict:
    """Best-effort "epsilon=0.1,lr=0.001" -> {epsilon: "0.1", lr: "0.001"}.

    The slug already carries the swept overrides (leaf names); this is only for
    display in results.json, so a value stays a string -- no re-typing games."""
    if not slug or slug == "base":
        return {}
    out: dict[str, str] = {}
    for part in slug.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def build_groups(exp: Experiment) -> list[GroupResult]:
    """Read every job's metrics.jsonl and reduce to one `GroupResult` per
    (experiment, env), in expand()'s experiment -> env emission order."""
    buckets: dict[tuple[str, str], dict] = {}
    for job in expand(exp):
        slug = job.label.split("|")[0]
        key = (slug, job.env)
        b = buckets.get(key)
        if b is None:
            b = {"slug": slug, "env": job.env,
                 "overrides": _parse_slug(slug), "jobs": []}
            buckets[key] = b
        b["jobs"].append(job)

    groups: list[GroupResult] = []
    for b in buckets.values():
        seeds = [seed_result(job.seed, Path(job.run_dir) / "metrics.jsonl")
                 for job in b["jobs"]]
        groups.append(group_result(b["slug"], b["env"], seeds,
                                    overrides=b["overrides"]))
    return groups


# --------------------------------------------------------------------------- #
#  ranking + the three artifacts
# --------------------------------------------------------------------------- #


def _rank_key(g: GroupResult):
    """Sort key: best final_eval first; groups with no eval (NaN) sink last."""
    m = g.final_eval.mean
    return (0, -m) if math.isfinite(m) else (1, 0.0)


def _ranking(groups: list[GroupResult]) -> list[dict]:
    ordered = sorted(groups, key=_rank_key)
    return [{"rank": i, "experiment": g.label, "env": g.env,
             "final_eval_mean": g.final_eval.mean,
             "final_eval_std": g.final_eval.std,
             "rew_last10_mean": g.rew_last10.mean,
             "n_seeds": g.final_eval.n}
            for i, g in enumerate(ordered, 1)]


def _results_dict(exp: Experiment, groups: list[GroupResult]) -> dict:
    return {"experiment": exp.name, "run_dir": str(exp.run_dir),
            "envs": list(exp.envs), "runs": exp.runs,
            "groups": [g.as_dict() for g in groups],
            "ranking": _ranking(groups)}


def _fmt(stat) -> str:
    return "n/a" if not math.isfinite(stat.mean) else f"{stat.mean:.1f} ± {stat.std:.1f}"


def write_summary_md(exp: Experiment, groups: list[GroupResult], path: Path) -> None:
    """One markdown section per env, experiments ranked by final_eval mean."""
    lines = [f"# {exp.name}", "",
             f"- run_dir: `{exp.run_dir}`",
             f"- runs (seeds 0..{exp.runs - 1}): {exp.runs}",
             f"- envs: {', '.join(exp.envs)}", ""]
    for env in exp.envs:
        env_groups = sorted((g for g in groups if g.env == env), key=_rank_key)
        lines += [f"## {env}", "",
                  "| rank | experiment | final_eval | rew_last10 | rew_last30 | seeds |",
                  "|---:|:---|:---|:---|:---|---:|"]
        for i, g in enumerate(env_groups, 1):
            lines.append(f"| {i} | {g.label} | {_fmt(g.final_eval)} | "
                         f"{_fmt(g.rew_last10)} | {_fmt(g.rew_last30)} | {g.final_eval.n} |")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def plot_curves(groups: list[GroupResult], run_dir: Path,
                log: Logger | None = None) -> list[Path]:
    """One reward-curve figure per env (a line per experiment, cross-seed mean
    with a +/- std band). OPTIONAL: if matplotlib is not installed the figures
    are skipped with a warning -- never an error."""
    try:
        import matplotlib
        matplotlib.use("Agg")                       # headless, no display needed
        import matplotlib.pyplot as plt
    except Exception as e:                           # noqa: BLE001 -- optional cap
        if log is not None:
            log.warning(f"plot skipped (matplotlib unavailable): {e}")
        return []

    envs: list[str] = []
    for g in groups:                                 # preserve order, unique
        if g.env not in envs:
            envs.append(g.env)

    written: list[Path] = []
    for env in envs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        drew = False
        for g in (x for x in groups if x.env == env):
            by_step: dict[int, list[float]] = defaultdict(list)
            for s in g.seeds:
                for step, y in s.curve:
                    by_step[step].append(y)
            if not by_step:
                continue
            steps = sorted(by_step)
            mean = [statistics.fmean(by_step[s]) for s in steps]
            std = [statistics.pstdev(by_step[s]) if len(by_step[s]) > 1 else 0.0
                   for s in steps]
            line, = ax.plot(steps, mean, label=g.label)
            ax.fill_between(steps, [m - d for m, d in zip(mean, std)],
                            [m + d for m, d in zip(mean, std)],
                            alpha=0.2, color=line.get_color())
            drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set_xlabel("global_step")
        ax.set_ylabel("rollout/ep_rew_mean")
        ax.set_title(env)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = Path(run_dir) / f"curve_{_env_slug(env)}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(out)
        if log is not None:
            log.info("curve written", path=str(out))
    return written


# --------------------------------------------------------------------------- #
#  the entry point
# --------------------------------------------------------------------------- #


def report(experiment: str | Path | Experiment, log: Logger | None = None) -> dict:
    """Build the three-piece for a (finished) sweep and write it to run_dir root.

    Accepts an already-parsed `Experiment` (what the runner passes) or a path to
    an experiment file (standalone re-report, no retraining). Returns the
    results.json dict."""
    exp = experiment if isinstance(experiment, Experiment) else load_experiment(experiment)
    _log = log or Logger()
    groups = build_groups(exp)
    run_dir = Path(exp.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "results.json").write_text(
        json.dumps(_results_dict(exp, groups), indent=2, default=str),
        encoding="utf-8")
    write_summary_md(exp, groups, run_dir / "summary.md")
    plot_curves(groups, run_dir, log=_log)

    _log.success("report written", groups=len(groups),
                 results=str(run_dir / "results.json"),
                 summary=str(run_dir / "summary.md"))
    return _results_dict(exp, groups)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="oprl results report (L4 stage 4)")
    ap.add_argument("experiment", help="path to the experiment file that was run")
    args = ap.parse_args()
    report(args.experiment)

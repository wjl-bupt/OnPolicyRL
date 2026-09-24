"""Experiment declaration + grid expansion -- L4 stage (1) declare (DESIGN_v2 §2).

An experiment file declares a fixed `baseline` config plus a `space` of
parameters to sweep, over an `envs` list and `runs` seeds. `expand()` takes the
Cartesian product of the space -- one "experiment" per combination -- and fans
each experiment across ALL envs x runs seeds, emitting jobs in the order
experiment -> env -> seed, so a sequential runner finishes one experiment
entirely (every env x run) before the next begins -- the framework's "排列组合".

Seeds are not declared as a list: `runs: N` pins them to the fixed sequence
0..N-1. Nobody can hand-pick lucky seeds -- an anti-cheating measure.

`run_dir` is normally NOT declared: the runner derives it as `runs/<name>` (a
deterministic root, so reruns reuse it and the model pool accumulates / resume
finds it). An explicit `run_dir` overrides, for debugging.

    baseline: fixed values (including list-valued ones like net_arch: [64, 64])
    space:    parameters to sweep, each mapped to a LIST of candidate values

Ambiguity "is this list a value or a sweep?" is resolved by POSITION, not by a
marker: a list in `baseline` is one fixed value; a list in `space` is the set of
candidates for that key. Keys in `space` are dotted paths into the resolved
config (e.g. `algo_params.epsilon`) because YAML does not auto-nest dotted keys.

The file is a thin front-end: the resolved config is a plain dict, so it hashes
cleanly for job identity and the future search stage (5) can emit the same dict
programmatically -- no file format in the loop. Format is chosen by suffix:
.yaml/.yml -> pyyaml (a core dep), .json -> json, .toml -> tomllib (3.11+). The
product logic below is ours; the file only carries data.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ALLOWED_TOP = {"algo", "env", "envs", "runs", "resume", "run_dir", "name",
                "baseline", "space"}


@dataclass(frozen=True)
class Job:
    """One resolved run: a single config, seed, and run directory."""

    algo: str
    env: str
    seed: int
    config: dict           # resolved {algo_params, network_params, monitor, ...}
    run_dir: Path
    resume: bool
    identity: str          # 12-hex sha1 of (algo, env, seed, resolved config)
    label: str             # human-readable "slug|env|seed-N" (logs + dir names)


@dataclass
class Experiment:
    """A parsed experiment file, before expansion."""

    algo: str
    envs: list[str]
    runs: int
    resume: bool
    run_dir: Path
    baseline: dict
    space: dict
    name: str


# --------------------------------------------------------------------------- #
#  loading
# --------------------------------------------------------------------------- #


def _load_raw(path: str | Path) -> Any:
    p = Path(path)
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".toml":
        import tomllib
        return tomllib.loads(text)
    raise ValueError(f"unsupported experiment format {suffix!r}; "
                     f"use .yaml / .yml / .json / .toml")


def load_experiment(path: str | Path) -> Experiment:
    """Parse an experiment file into an `Experiment` (no expansion yet).

    Unknown top-level keys raise here, at load time -- a typo in the experiment
    file should fail before a single job starts, not silently do nothing."""
    data = _load_raw(path)
    if not isinstance(data, dict):
        raise ValueError("experiment file must be a mapping at the top level")
    unknown = set(data) - _ALLOWED_TOP
    if unknown:
        raise ValueError(f"unknown top-level keys {sorted(unknown)}; "
                         f"allowed keys are {sorted(_ALLOWED_TOP)}")
    if "algo" not in data:
        raise ValueError("experiment must declare 'algo'")

    # env axis: `envs` (a list) is canonical; `env` (a scalar) is 1-element sugar.
    if "envs" in data:
        envs = data["envs"]
        if not isinstance(envs, (list, tuple)) or not envs:
            raise ValueError("'envs' must be a non-empty list of environment ids")
        envs = [str(e) for e in envs]
    elif "env" in data:
        envs = [str(data["env"])]
    else:
        raise ValueError("experiment must declare 'envs' (a list) or 'env'")

    # seed axis: `runs: N` runs the FIXED seeds 0..N-1 -- there is no `seeds` list,
    # so nobody can hand-pick lucky seeds (an anti-cheating measure).
    runs = int(data.get("runs", 1))
    if runs < 1:
        raise ValueError("'runs' must be >= 1")

    name = str(data.get("name") or Path(path).stem)
    # run_dir is DERIVED from `name` (deterministic runs/<name>) so a sweep reuses
    # one root across reruns -- the model pool accumulates there and resume finds
    # it. An explicit `run_dir` overrides (debugging); files normally omit it. NOT
    # timestamped on purpose: a per-launch root would defeat resume / accumulation.
    run_dir = Path(data["run_dir"]) if data.get("run_dir") else Path("runs") / name
    return Experiment(
        algo=str(data["algo"]),
        envs=envs,
        runs=runs,
        resume=bool(data.get("resume", False)),
        run_dir=run_dir,
        baseline=dict(data.get("baseline") or {}),
        space=dict(data.get("space") or {}),
        name=name,
    )


# --------------------------------------------------------------------------- #
#  expansion
# --------------------------------------------------------------------------- #


def _set_by_path(node: dict, dotted: str, value: Any) -> None:
    """Set a nested key addressed by a dotted path, creating dicts on the way."""
    keys = dotted.split(".")
    for k in keys[:-1]:
        nxt = node.get(k)
        if nxt is None:
            nxt = {}
            node[k] = nxt
        elif not isinstance(nxt, dict):
            raise ValueError(f"override path {dotted!r} runs through a non-dict "
                             f"value at {k!r}")
        node = nxt
    node[keys[-1]] = value


def _slug(overrides: dict) -> str:
    """A filesystem-safe label for one swept combination (dir name + log tag)."""
    if not overrides:
        return "base"
    parts = [f"{k.split('.')[-1]}={v}" for k, v in overrides.items()]
    raw = ",".join(parts)
    return "".join(c if (c.isalnum() or c in "=,.-_") else "_" for c in raw)


def _env_slug(env: str) -> str:
    """Filesystem-safe directory name for an env id (MinAtar/Breakout-v0 -> ...-...)."""
    return "".join(c if (c.isalnum() or c in ".-_") else "-" for c in env)


def _identity(algo: str, env: str, seed: int, config: dict) -> str:
    blob = json.dumps({"algo": algo, "env": env, "seed": seed, "config": config},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def expand(exp: Experiment) -> list[Job]:
    """The Cartesian product of `exp.space` -> one experiment (param combo) each;
    every experiment then fans across ALL `envs` x `runs` seeds.

    Jobs are emitted in the order experiment -> env -> seed, so a sequential
    runner finishes every env x run of one experiment before the next experiment
    starts. Empty space -> one experiment (the pure baseline). Each `space` value
    MUST be a list of candidates; a scalar there is a mistake (it belongs in the
    baseline) and raises."""
    keys = list(exp.space.keys())
    for k in keys:
        if not isinstance(exp.space[k], (list, tuple)):
            raise ValueError(
                f"space[{k!r}] must be a list of candidate values to sweep "
                f"(a scalar is a fixed value -- put it in `baseline`)")
    value_lists = [list(exp.space[k]) for k in keys]

    jobs: list[Job] = []
    for combo in itertools.product(*value_lists):            # each combo = one experiment
        overrides = dict(zip(keys, combo))
        resolved_base = copy.deepcopy(exp.baseline)
        for dotted, val in overrides.items():
            _set_by_path(resolved_base, dotted, val)
        slug = _slug(overrides)
        for env in exp.envs:                                 # all envs of this experiment
            env_dir = _env_slug(env)
            for seed in range(exp.runs):                     # fixed seeds 0..runs-1
                config = copy.deepcopy(resolved_base)
                jobs.append(Job(
                    algo=exp.algo, env=env, seed=seed, config=config,
                    run_dir=exp.run_dir / slug / env_dir / f"seed-{seed}",
                    resume=exp.resume,
                    identity=_identity(exp.algo, env, seed, config),
                    label=f"{slug}|{env}|seed-{seed}",
                ))
    return jobs

"""Config base class and YAML loading -- every hyperparameter is listed explicitly,
with no defaults hidden elsewhere.

Precedence (later wins)::

    dataclass defaults  <  a section of config/<algo>.yaml  <  CLI arguments

Unknown keys always raise. A misspelled hyperparameter in a config file must be
immediately visible rather than silently ignored.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

# The repo-root config/ directory; falls back to CWD/config when installed as a package.
_HERE = Path(__file__).resolve()
CONFIG_DIR = next(
    (p / "config" for p in _HERE.parents if (p / "config").is_dir()),
    Path.cwd() / "config",
)


@dataclass
class Config:
    """Hyperparameters shared by every algorithm; algorithm-specific ones live in
    the respective Config subclass."""

    seed: int = 1
    device: str = "auto"
    total_steps: int = 1_000_000
    num_envs: int = 8
    rollout_len: int = 128
    log_interval: int = 1
    run_dir: str | None = None
    smoke: bool = False
    # Force deterministic kernels. Off by default: it costs throughput and some ops
    # have no deterministic implementation.
    deterministic: bool = False

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def to_dict(self) -> dict[str, Any]:
        """For serialization. Runtime fields starting with `_` are excluded."""
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls, d: dict, strict: bool = True):
        """Build a config. With strict=True, unknown keys raise with a spelling hint."""
        known = cls.field_names()
        unknown = {k for k in d if k not in known and not k.startswith("_")}
        if unknown and strict:
            hints = []
            for k in sorted(unknown):
                near = _closest(k, known)
                hints.append(f"  {k!r}" + (f"  (did you mean {near!r}?)" if near else ""))
            raise ValueError(
                f"{cls.__name__} does not recognize these hyperparameters:\n"
                + "\n".join(hints)
                + f"\navailable: {sorted(known)}"
            )
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".yaml", ".yml"):
            path.write_text(dump_yaml(self.to_dict()), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def describe(self) -> str:
        """Print the resolved config, marking values that deviate from defaults."""
        defaults = type(self)()
        lines = [f"{type(self).__name__}:"]
        for f in sorted(fields(self), key=lambda x: x.name):
            if f.name.startswith("_"):
                continue
            v, dv = getattr(self, f.name), getattr(defaults, f.name)
            mark = "  *" if v != dv else ""
            lines.append(f"  {f.name:<20} {v!r}{mark}")
        if any("*" in ln for ln in lines):
            lines.append("  (* = differs from default)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Preset loading: one section of config/<algo>.yaml
# --------------------------------------------------------------------------- #
#
# Design choice: **one file per algorithm, one section per environment family**.
# An earlier version used one YAML per experiment, which produced 15 files of which 12
# were five-line stubs (a whole file just to set `surrogate: dpo`). Nobody wants to read
# that. If a change fits in one command-line flag, it does not need a config file.


def _read(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading YAML requires pyyaml: uv add pyyaml") from e
    return yaml.safe_load(text) or {}


def algo_file(algo: str) -> Path:
    for suf in (".yaml", ".yml", ".json"):
        p = CONFIG_DIR / f"{algo}{suf}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"no such file: {CONFIG_DIR}/{algo}.yaml")


def load_dict(preset: str | None, algo: str = "ppo") -> dict:
    """Read the section named `preset` from config/<algo>.yaml as a flat dict.

    `preset` may also be a path to a YAML/JSON file, which is how a saved run config is
    replayed. The `note` key is prose and never becomes a hyperparameter.
    """
    if not preset or preset == "default":
        if preset == "default":
            return _section(algo, "default")
        return {}
    # Paths take priority, enabling `--config runs/xxx/config.yaml` replays.
    p = Path(preset)
    if p.is_file():
        d = _read(p)
        d.pop("note", None)
        return d
    return _section(algo, preset)


def _section(algo: str, name: str) -> dict:
    doc = _read(algo_file(algo))
    if name not in doc:
        raise KeyError(
            f"config/{algo}.yaml has no preset {name!r}; "
            f"available: {[k for k in doc if not k.startswith('_')]}"
        )
    sec = dict(doc[name] or {})
    sec.pop("note", None)   # prose, not a hyperparameter
    return sec


def load_config(cls, preset: str | None = None, algo: str = "ppo", **overrides):
    """Build a config from a preset plus keyword overrides.

    Example::

        cfg = load_config(PPOConfig, "mujoco", lr=1e-4, seed=3)
    """
    d = load_dict(preset, algo)
    d.update({k: v for k, v in overrides.items() if v is not None})
    return cls.from_dict(d)


def presets(algo: str = "ppo") -> dict[str, str]:
    """List an algorithm's presets together with their notes."""
    try:
        doc = _read(algo_file(algo))
    except FileNotFoundError:
        return {}
    return {
        k: str((v or {}).get("note", "")).strip()
        for k, v in doc.items()
        if not k.startswith("_")
    }


# experiments.yaml describes experiment orchestration, not algorithm hyperparameters.
NON_ALGO_FILES = {"experiments"}


def algo_names() -> list[str]:
    """Algorithm config file stems under config/ (excluding experiments.yaml)."""
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(
        f.stem for f in CONFIG_DIR.glob("*.y*ml") if f.stem not in NON_ALGO_FILES
    )


def available() -> list[str]:
    """Every (algo, preset) pair, for CLI listings."""
    out = []
    for algo in algo_names():
        out += [f"{algo}:{k}" for k in presets(algo)]
    return out


def dump_yaml(d: dict, indent: int = 0) -> str:
    """Minimal YAML writer covering only the scalars and nested dicts a config uses.

    Keeps the write path free of the yaml dependency, so saving a resolved config works
    in any environment.
    """
    pad = "  " * indent
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(dump_yaml(v, indent + 1))
        elif v is None:
            lines.append(f"{pad}{k}: null")
        elif isinstance(v, bool):
            lines.append(f"{pad}{k}: {str(v).lower()}")
        elif isinstance(v, str):
            lines.append(f"{pad}{k}: {v}")
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


def _closest(word: str, candidates: set[str]) -> str | None:
    """Spelling hints via stdlib; no extra dependency."""
    import difflib

    m = difflib.get_close_matches(word, sorted(candidates), n=1, cutoff=0.6)
    return m[0] if m else None


__all__ = [
    "Config", "load_config", "load_dict", "presets", "available", "algo_names",
    "algo_file", "dump_yaml", "CONFIG_DIR",
]

"""Architecture discipline tests -- the automated half of DESIGN.md §4.7.

These are not functional tests; they are guard rails that stop the design from rotting.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "oprl"

# The L1 primitive layer: none of these may import algos/.
L1 = [
    "types.py", "schema.py", "buffer.py", "rollout.py", "logger.py",
    "norm.py", "config.py", "metrics.py", "tree.py", "registry.py",
    "advantages", "objectives", "nets", "envs",
]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
        elif isinstance(node, ast.Import):
            out += [a.name for a in node.names]
    return out


def _py_files(name: str) -> list[Path]:
    p = SRC / name
    return sorted(p.rglob("*.py")) if p.is_dir() else [p]


@pytest.mark.parametrize("target", L1)
def test_l1_does_not_import_algos(target):
    """**Core discipline**: L1 depends one way only and never imports algos/.

    The moment a reverse dependency appears, the primitives are contaminated by an
    algorithm and `from oprl import gae` stops being independently usable
    (DESIGN.md §4.7, rule 1).
    """
    for f in _py_files(target):
        for mod in _imports(f):
            assert "algos" not in mod, f"{f.relative_to(SRC)} imports algos: {mod}"


@pytest.mark.parametrize("target", L1)
def test_l1_does_not_import_cli(target):
    """L1/L2 must not import the L3 convenience layer."""
    for f in _py_files(target):
        for mod in _imports(f):
            assert not mod.endswith("cli"), f"{f.relative_to(SRC)} imports cli"


def test_types_only_depends_on_torch():
    """L0 depends on torch alone -- it is the foundation and may have no in-package deps."""
    for mod in _imports(SRC / "types.py"):
        assert not mod.startswith("."), f"types.py must have no in-package deps: {mod}"


def test_core_deps_only():
    """Dependency budget: at module level, src/ may import only torch, numpy, gymnasium
    and the stdlib.

    Optional backends (tensorboard / wandb / matplotlib / env suites) must be imported
    inside functions, so `import oprl` never pulls them in (DESIGN.md §4.8).
    """
    # pyyaml is the fourth core dependency (configuration is a core feature and cannot
    # hide behind an extra), but config.py still imports it lazily because .json configs
    # do not need it.
    banned = {"tensorboard", "wandb", "matplotlib", "pandas", "minatar",
              "mujoco", "ale_py", "minigrid", "tensordict", "protobuf", "yaml"}
    for f in sorted(SRC.rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # Only module-level imports are checked.
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node.col_offset == 0:
                mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                        else [node.module or ""])
                for m in mods:
                    root = m.split(".")[0]
                    assert root not in banned, (
                        f"{f.relative_to(SRC)} imports optional dependency {root} at module "
                        "level; move it inside a function"
                    )


def test_protocol_method_budget():
    """Each protocol keeps to a small method budget; exceeding it means the abstraction
    is wrong (DESIGN.md §4.7, rule 2)."""
    limits = {"EnvAdapter": 5, "Policy": 5, "Sink": 5, "Surrogate": 5,
              "AdvantageEstimator": 6}
    for f in sorted(SRC.rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in limits:
                n = sum(1 for b in node.body if isinstance(b, ast.FunctionDef))
                assert n <= limits[node.name], (
                    f"{node.name} has {n} methods, over its budget of {limits[node.name]}"
                )


def test_cli_imports_resolve():
    """The CLI's lazy imports must also resolve.

    This test exists because a directory reshuffle broke cli.py's `from .policy import`
    while every other test still passed: lazy imports are not covered by normal tests.
    """
    import importlib

    from oprl import cli

    src = Path(cli.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            importlib.import_module(f"oprl.{node.module}")


def test_config_matches_dataclasses():
    """Every key in config/*.yaml must be a real field of the matching Config.

    Prevents **config drift**: renaming a dataclass field but not the YAML, or vice versa.
    """
    from oprl.algos.ppo import PPOConfig
    from oprl.algos.vmpo import VMPOConfig
    from oprl.config import available, load_dict

    for entry in available():
        algo, name = entry.split(":", 1)
        cls = VMPOConfig if algo == "vmpo" else PPOConfig
        unknown = set(load_dict(name, algo)) - cls.field_names()
        assert not unknown, f"{entry} has keys unknown to {cls.__name__}: {sorted(unknown)}"


def test_config_files_stay_small():
    """Config size budget: **one file per algorithm**, never back to one-file-per-experiment.

    History: this was once 15 files and 252 lines, of which 12 were five-line stubs
    created just to change a single field.
    """
    cfg_dir = SRC.parents[1] / "config"
    files = sorted(cfg_dir.glob("*.y*ml"))
    assert len(files) <= 5, f"config/ has {len(files)} files; group them by algorithm"
    total = sum(len(f.read_text(encoding="utf-8").splitlines()) for f in files)
    assert total <= 250, f"config/ totals {total} lines, over budget"

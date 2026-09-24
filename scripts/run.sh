#!/usr/bin/env bash
# oprl experiment launcher (L4 stage (2) run) -- the official entry point.
#
#   scripts/run.sh <experiment.yaml|.json|.toml> [--device cpu]
#
# It resolves the repo, picks the project interpreter (.venv if present), and
# hands the config path to the Python runner. The runner writes a verbatim
# snapshot of the config under the experiment's run_dir, so the exact
# declaration that produced a sweep is always archived (each job's config.json
# holds the RESOLVED config; the snapshot is the raw source).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: scripts/run.sh <experiment.yaml|.json|.toml> [--device DEV]" >&2
    exit 2
fi

# repo root = the parent of this script's directory (scripts/..)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="$1"; shift
if [[ ! -f "$CONFIG" ]]; then
    echo "oprl: config not found: $CONFIG" >&2
    exit 2
fi

# prefer the project venv interpreter; fall back to whatever python is on PATH
PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python"

# src on PYTHONPATH so the namespace packages import cleanly (mirrors conftest;
# runner.py also self-inserts it when run as __main__ -- belt and suspenders)
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PY" "$REPO_ROOT/src/workflow/runner.py" "$CONFIG" "$@"

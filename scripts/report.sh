#!/usr/bin/env bash
# oprl results report (L4 stage (4)) -- regenerate the results three-piece
# (results.json + summary.md + per-env curve PNGs) from a FINISHED sweep without
# retraining. It reads the run_dir named by the same experiment file you ran.
#
#   scripts/report.sh <experiment.yaml|.json|.toml>
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: scripts/report.sh <experiment.yaml|.json|.toml>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="$1"; shift
if [[ ! -f "$CONFIG" ]]; then
    echo "oprl: config not found: $CONFIG" >&2
    exit 2
fi

PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PY" "$REPO_ROOT/src/workflow/report.py" "$CONFIG" "$@"

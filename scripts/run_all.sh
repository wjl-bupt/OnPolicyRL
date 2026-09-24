#!/usr/bin/env bash
# oprl one-shot: TRAIN then REPORT from a single config -- the whole loop in one
# command. runner.py already writes the report at the end of a sweep; this re-runs
# report.py afterwards (idempotent) so the results three-piece is refreshed even
# if the in-runner report was skipped, and gives you a single entry point.
#
#   scripts/run_all.sh <experiment.yaml|.json|.toml> [runner args, e.g. --device cpu]
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: scripts/run_all.sh <experiment.yaml|.json|.toml> [runner args...]" >&2
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

echo "[run_all] training  : $CONFIG"
"$PY" "$REPO_ROOT/src/workflow/runner.py" "$CONFIG" "$@"     # train (writes report)
echo "[run_all] reporting : $CONFIG"
"$PY" "$REPO_ROOT/src/workflow/report.py" "$CONFIG"          # refresh the three-piece
echo "[run_all] done -- results under the run_dir logged above."

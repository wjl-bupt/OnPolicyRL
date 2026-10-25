#!/usr/bin/env bash
# Re-plot saved data; does not resume any experiment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CAMPAIGN="${CAMPAIGN:-runs/jax_minatar_5seed_10m/campaign_20261024}"
OUTPUT="${OUTPUT:-runs/jax_minatar_5seed_10m/report_stopped_20261025}"
exec "${PYTHON:-$ROOT/.venv-jax/bin/python}" scripts/plot_campaign.py "$CAMPAIGN" --output "$OUTPUT" "$@"

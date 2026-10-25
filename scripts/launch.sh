#!/usr/bin/env bash
# Edit this parameter section, or set environment variables before invoking.
# Default is preview. Pass --run to actually start the jobs, --prepare to snapshot.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${CONFIG:-configs/minatar_jax.yaml}"
PYTHON="${PYTHON:-$ROOT/.venv-jax/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
COMMIT="${COMMIT:-}"                    # Empty: current checkout; otherwise exact commit.
ALGOS="${ALGOS:-[ppo, ga2e]}"           # YAML list or one name, e.g. ppo
ENVS="${ENVS:-}"                       # Empty: config list. E.g. '[MinAtar/Asterix-v0]'
RUNS="${RUNS:-5}"
TOTAL_STEPS="${TOTAL_STEPS:-10000000}"
ARCH="${ARCH:-}"                       # Empty: config; mlp/simba/bronet/spectralnet/causal_transformer
ACTION_HEAD="${ACTION_HEAD:-}"         # auto or diffusion (requires ALGOS=dppo)
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"   # causal_transformer observation window
DIFFUSION_STEPS="${DIFFUSION_STEPS:-}" # number of Gaussian denoising transitions
# Put extra per-algorithm / per-environment / wrapper settings here.
OVERRIDES=(
  "algo=$ALGOS"
  "runs=$RUNS"
  "train.total_steps=$TOTAL_STEPS"
  # 'algorithms.ppo.train.gae_lambda=0.95'
  # 'algorithms.ga2e.train.lambda_val=0.99'
  # 'environment.sticky_action_prob=0.1'   # MinAtar only
  # 'environment.jax_platform=cpu'
  # 'model.hidden_dim=256'
  # 'model.num_blocks=2'
  # 'model.lr=0.0001'
)
[[ -z "$ENVS" ]] || OVERRIDES+=("envs=$ENVS")
[[ -z "$ARCH" ]] || OVERRIDES+=("model.architecture=$ARCH")
[[ -z "$ACTION_HEAD" ]] || OVERRIDES+=("model.action_head=$ACTION_HEAD")
[[ -z "$CONTEXT_LENGTH" ]] || OVERRIDES+=("model.context_length=$CONTEXT_LENGTH")
[[ -z "$DIFFUSION_STEPS" ]] || OVERRIDES+=("model.diffusion_steps=$DIFFUSION_STEPS")
MODE=plan
if [[ "${1:-}" == --run ]]; then MODE=run; shift
elif [[ "${1:-}" == --prepare ]]; then MODE=prepare; shift
elif [[ "${1:-}" == --plan ]]; then shift
fi
ARGS=("$CONFIG" --device "$DEVICE")
for option in "${OVERRIDES[@]}"; do ARGS+=(--set "$option"); done
if [[ "$MODE" == plan ]]; then
  ARGS+=(--plan)
  [[ -z "$COMMIT" ]] || echo 'Preview uses current code; use --prepare to inspect pinned code setup.' >&2
else
  [[ -z "$COMMIT" ]] || ARGS+=(--commit "$COMMIT" --repo "$ROOT")
  if [[ "$MODE" == prepare ]]; then
    [[ -n "$COMMIT" ]] || { echo '--prepare requires COMMIT' >&2; exit 2; }
    ARGS+=(--prepare-only)
  fi
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec "$PYTHON" -m oprl.cli "${ARGS[@]}" "$@"

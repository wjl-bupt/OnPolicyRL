#!/usr/bin/env bash
# Run SPO, GA2E, PPO on MinAtar/Breakout-v1, 10M steps each, CPU, in order.
set -euo pipefail
cd /data/workspace/OnPolicyRL

run() {
  local name="$1"; shift
  local out="runs/${name}-MinAtar-Breakout-v1-seed1.stdout.log"
  echo "=== [$(date +%H:%M:%S)] START ${name} -> ${out} ==="
  uv run oprl train ppo --config "$@" --env MinAtar/Breakout-v1 \
    --total_steps 10000000 --device cpu --seed 1 \
    --run_dir "runs/${name}-MinAtar-Breakout-v1-seed1" \
    > "$out" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE  ${name} (exit $?) ==="
}

run spo-minatar  minatar --surrogate spo
run ga2e-minatar .runtmp/ga2e_minatar.yaml
run ppo-minatar  minatar

echo "ALL DONE"

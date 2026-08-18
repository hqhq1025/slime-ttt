#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# AC1 completed rollout ids 0..2 before a direct-convolution verifier stall.
# Resume from its persisted archive/best state for the remaining two updates.
echo "[TTT suite] resuming ac1 at $(date -u +%FT%TZ)"
# slime interprets --num-rollout as the exclusive end index, not a count.
TTT_RUN_NAME=qwen3-4b-ttt-ac1-8x16x5 \
TTT_START_ROLLOUT=3 \
TTT_NUM_ROLLOUT=5 \
  "${LOCAL_DIR}/run_ttt_domain_scale.sh" ac1
echo "[TTT suite] completed ac1 at $(date -u +%FT%TZ)"

for task in ac2 circle26 circle32; do
  echo "[TTT suite] starting ${task} at $(date -u +%FT%TZ)"
  "${LOCAL_DIR}/run_ttt_domain_scale.sh" "${task}"
  echo "[TTT suite] completed ${task} at $(date -u +%FT%TZ)"
done

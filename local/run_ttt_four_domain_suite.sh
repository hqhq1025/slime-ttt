#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for task in ac1 ac2 circle26 circle32; do
  echo "[TTT suite] starting ${task} at $(date -u +%FT%TZ)"
  "${LOCAL_DIR}/run_ttt_domain_scale.sh" "${task}"
  echo "[TTT suite] completed ${task} at $(date -u +%FT%TZ)"
done

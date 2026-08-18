#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TTT_RUN_NAME="${TTT_RUN_NAME:-qwen3-4b-ttt-trimul-a100-lora-r32-8x8x3}"
export TTT_NUM_ROLLOUT="${TTT_NUM_ROLLOUT:-3}"
export TTT_ROLLOUT_BATCH_SIZE="${TTT_ROLLOUT_BATCH_SIZE:-1}"
export TTT_SAMPLES_PER_PROMPT="${TTT_SAMPLES_PER_PROMPT:-8}"
export TTT_GLOBAL_BATCH_SIZE="${TTT_GLOBAL_BATCH_SIZE:-8}"
export TTT_SAVE_INTERVAL="${TTT_SAVE_INTERVAL:-3}"
export TTT_LORA_RANK="${TTT_LORA_RANK:-32}"
export TTT_LORA_ALPHA="${TTT_LORA_ALPHA:-32}"
export TTT_LORA_DROPOUT="${TTT_LORA_DROPOUT:-0.0}"
export TTT_LR="${TTT_LR:-4e-5}"
exec "${LOCAL_DIR}/run_ttt_trimul_a100_smoke.sh"

#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Match the official adapter capacity while keeping the rollout/verifier setup
# identical to the full-parameter Qwen3-4B smoke.
export TTT_RUN_NAME="${TTT_RUN_NAME:-qwen3-4b-ttt-erdos-lora-r32-smoke}"
export TTT_NUM_ROLLOUT="${TTT_NUM_ROLLOUT:-1}"
export TTT_ROLLOUT_BATCH_SIZE="${TTT_ROLLOUT_BATCH_SIZE:-1}"
export TTT_SAMPLES_PER_PROMPT="${TTT_SAMPLES_PER_PROMPT:-4}"
export TTT_GLOBAL_BATCH_SIZE="${TTT_GLOBAL_BATCH_SIZE:-4}"

export TTT_LORA_RANK="${TTT_LORA_RANK:-32}"
export TTT_LORA_ALPHA="${TTT_LORA_ALPHA:-32}"
export TTT_LORA_DROPOUT="${TTT_LORA_DROPOUT:-0.0}"
export TTT_LR="${TTT_LR:-4e-5}"
export TTT_SGLANG_DISABLE_CUDA_GRAPH="${TTT_SGLANG_DISABLE_CUDA_GRAPH:-1}"

exec "${LOCAL_DIR}/run_ttt_erdos_smoke.sh"

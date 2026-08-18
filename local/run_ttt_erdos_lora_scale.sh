#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Same online-search budget as qwen3-4b-ttt-erdos-8x16x10; only the adaptation
# method and its method-appropriate learning rate differ.
export TTT_RUN_NAME="${TTT_RUN_NAME:-qwen3-4b-ttt-erdos-lora-r32-8x16x10}"
export TTT_NUM_ROLLOUT="${TTT_NUM_ROLLOUT:-10}"
export TTT_ROLLOUT_BATCH_SIZE="${TTT_ROLLOUT_BATCH_SIZE:-8}"
export TTT_SAMPLES_PER_PROMPT="${TTT_SAMPLES_PER_PROMPT:-16}"
export TTT_GLOBAL_BATCH_SIZE="${TTT_GLOBAL_BATCH_SIZE:-128}"
export TTT_RESPONSE_LEN="${TTT_RESPONSE_LEN:-8192}"
export TTT_PHASE1_CONTEXT="${TTT_PHASE1_CONTEXT:-4096}"
export TTT_MAX_TOKENS_PER_GPU="${TTT_MAX_TOKENS_PER_GPU:-8192}"
export TTT_EVAL_CONCURRENCY="${TTT_EVAL_CONCURRENCY:-32}"
export TTT_MAX_BUFFER_SIZE="${TTT_MAX_BUFFER_SIZE:-1000}"
export TTT_SAVE_INTERVAL="${TTT_SAVE_INTERVAL:-5}"

export TTT_LORA_RANK="${TTT_LORA_RANK:-32}"
export TTT_LORA_ALPHA="${TTT_LORA_ALPHA:-32}"
export TTT_LORA_DROPOUT="${TTT_LORA_DROPOUT:-0.0}"
export TTT_LR="${TTT_LR:-4e-5}"
export TTT_SGLANG_DISABLE_CUDA_GRAPH="${TTT_SGLANG_DISABLE_CUDA_GRAPH:-1}"

exec "${LOCAL_DIR}/run_ttt_erdos_smoke.sh"

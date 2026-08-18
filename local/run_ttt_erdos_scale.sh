#!/usr/bin/env bash
set -euo pipefail

# Staged scale run: 8 independent PUCT groups x 16 samples x 10 updates.
# Every setting can still be overridden from the environment.
export TTT_RUN_NAME="${TTT_RUN_NAME:-qwen3-4b-ttt-erdos-8x16x10}"
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

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_ttt_erdos_smoke.sh"

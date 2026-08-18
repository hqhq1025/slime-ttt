#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TTT_ENV_PATH="examples.ttt_discover.envs.denoising.DenoisingPancreasEnv"
export TTT_TARGET="${TTT_TARGET:-0.15}"
export TTT_RUN_NAME="${TTT_RUN_NAME:-qwen3-4b-ttt-denoising-pancreas-smoke}"
export TTT_NUM_ROLLOUT="${TTT_NUM_ROLLOUT:-1}"
export TTT_ROLLOUT_BATCH_SIZE="${TTT_ROLLOUT_BATCH_SIZE:-1}"
export TTT_SAMPLES_PER_PROMPT="${TTT_SAMPLES_PER_PROMPT:-4}"
export TTT_GLOBAL_BATCH_SIZE="${TTT_GLOBAL_BATCH_SIZE:-4}"
export TTT_RESPONSE_LEN="${TTT_RESPONSE_LEN:-16384}"
export TTT_PHASE1_CONTEXT="${TTT_PHASE1_CONTEXT:-20480}"
export TTT_MAX_TOKENS_PER_GPU="${TTT_MAX_TOKENS_PER_GPU:-20480}"
export TTT_EVAL_TIMEOUT="${TTT_EVAL_TIMEOUT:-530}"
# A complex candidate peaks near 6.1 GB RSS.  Keep evaluations serial so a
# malformed generation cannot multiply its memory pressure fourfold.
export TTT_EVAL_CONCURRENCY=1
export TTT_NUM_CPUS_PER_TASK=2
export TTT_MAX_BUFFER_SIZE="${TTT_MAX_BUFFER_SIZE:-8}"

exec "${LOCAL_DIR}/run_ttt_erdos_smoke.sh"

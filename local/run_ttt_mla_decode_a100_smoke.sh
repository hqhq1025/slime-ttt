#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_DIR}/env.sh"
OFFICIAL_ROOT="${TTT_MLA_OFFICIAL_ROOT:-${TTT_OFFICIAL_ROOT}}"
RELEASED_CODE="${OFFICIAL_ROOT}/results/kernel-engineering/mla_code_3.py"

[[ -s "${RELEASED_CODE}" ]] || {
  echo "Missing released MLA Decode warm start: ${RELEASED_CODE}" >&2
  exit 2
}

export TTT_ENV_PATH="examples.ttt_discover.envs.mla_decode_a100.MLADecodeA100Env"
export TTT_TARGET="${TTT_TARGET:-500.0}"
export TTT_RUN_NAME="${TTT_RUN_NAME:-qwen3-4b-ttt-mla-decode-a100-smoke}"
export TTT_NUM_ROLLOUT="${TTT_NUM_ROLLOUT:-1}"
export TTT_ROLLOUT_BATCH_SIZE="${TTT_ROLLOUT_BATCH_SIZE:-1}"
export TTT_SAMPLES_PER_PROMPT="${TTT_SAMPLES_PER_PROMPT:-4}"
# The default 8-GPU layout has data-parallel size 4.
export TTT_GLOBAL_BATCH_SIZE="${TTT_GLOBAL_BATCH_SIZE:-4}"
export TTT_RESPONSE_LEN="${TTT_RESPONSE_LEN:-16384}"
export TTT_PHASE1_CONTEXT="${TTT_PHASE1_CONTEXT:-24576}"
export TTT_MAX_TOKENS_PER_GPU="${TTT_MAX_TOKENS_PER_GPU:-24576}"
# A checked released candidate takes ~52 seconds end-to-end.  Generated Triton
# candidates may compile more slowly, so retain ample per-candidate headroom.
export TTT_EVAL_TIMEOUT="${TTT_EVAL_TIMEOUT:-300}"
# Every evaluation is pinned to the same physical GPU; serial execution avoids
# four simultaneous ~12-GiB CUDA contexts colliding with the colocated model.
export TTT_EVAL_CONCURRENCY=1
export TTT_MAX_BUFFER_SIZE="${TTT_MAX_BUFFER_SIZE:-16}"
export TTT_MLA_REPEATS="${TTT_MLA_REPEATS:-3}"
export TTT_MLA_GPU_DEVICE="${TTT_MLA_GPU_DEVICE:-0}"
export TTT_MLA_OFFICIAL_ROOT="${OFFICIAL_ROOT}"
export TTT_MLA_WARM_START="${TTT_MLA_WARM_START:-1}"
export TTT_MLA_SEED_SCORE_US="${TTT_MLA_SEED_SCORE_US:-539.4624114971953}"

exec "${LOCAL_DIR}/run_ttt_erdos_smoke.sh"

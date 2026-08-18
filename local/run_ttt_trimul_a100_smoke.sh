#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TTT_ENV_PATH="examples.ttt_discover.envs.trimul_a100.TriMulA100Env"
export TTT_TARGET="${TTT_TARGET:-1000.0}"
export TTT_RUN_NAME="${TTT_RUN_NAME:-qwen3-4b-ttt-trimul-a100-smoke}"
export TTT_NUM_ROLLOUT="${TTT_NUM_ROLLOUT:-1}"
export TTT_ROLLOUT_BATCH_SIZE="${TTT_ROLLOUT_BATCH_SIZE:-1}"
export TTT_SAMPLES_PER_PROMPT="${TTT_SAMPLES_PER_PROMPT:-4}"
# The default 8-GPU layout has data-parallel size 4, so Megatron requires the
# global batch size to be divisible by 4.
export TTT_GLOBAL_BATCH_SIZE="${TTT_GLOBAL_BATCH_SIZE:-4}"
export TTT_RESPONSE_LEN="${TTT_RESPONSE_LEN:-16384}"
export TTT_PHASE1_CONTEXT="${TTT_PHASE1_CONTEXT:-24576}"
export TTT_MAX_TOKENS_PER_GPU="${TTT_MAX_TOKENS_PER_GPU:-24576}"
export TTT_EVAL_TIMEOUT="${TTT_EVAL_TIMEOUT:-300}"
export TTT_EVAL_CONCURRENCY=1
export TTT_MAX_BUFFER_SIZE="${TTT_MAX_BUFFER_SIZE:-16}"
export TTT_TRIMUL_PROFILE=smoke
export TTT_TRIMUL_REPEATS="${TTT_TRIMUL_REPEATS:-3}"
export TTT_TRIMUL_GPU_DEVICE="${TTT_TRIMUL_GPU_DEVICE:-0}"
# Official TriMul starts empty.  The 4B local smoke uses the released corrected
# A100 kernel as a valid parent so the run exercises kernel improvement.
export TTT_TRIMUL_WARM_START="${TTT_TRIMUL_WARM_START:-1}"
export TTT_TRIMUL_SEED_SCORE_US="${TTT_TRIMUL_SEED_SCORE_US:-903.62}"

# The TriMul evaluator itself uses one local A100.  Keeping candidate evaluation
# serial avoids CUDA-context OOMs while slime's colocated model is resident.
exec "${LOCAL_DIR}/run_ttt_erdos_smoke.sh"

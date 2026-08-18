#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?usage: run_ttt_domain_scale.sh ac1|ac2|circle26|circle32}"

case "${TASK}" in
  ac1)
    export TTT_ENV_PATH="examples.ttt_discover.envs.ac_inequalities.AC1Env"
    export TTT_TARGET="1.5030"
    ;;
  ac2)
    export TTT_ENV_PATH="examples.ttt_discover.envs.ac_inequalities.AC2Env"
    export TTT_TARGET="0.9700"
    ;;
  circle26)
    export TTT_ENV_PATH="examples.ttt_discover.envs.circle_packing.CirclePacking26Env"
    export TTT_TARGET="2.6360"
    ;;
  circle32)
    export TTT_ENV_PATH="examples.ttt_discover.envs.circle_packing.CirclePacking32Env"
    export TTT_TARGET="2.9400"
    ;;
  *)
    echo "unknown task: ${TASK}" >&2
    exit 2
    ;;
esac

# Medium local reproduction: 640 verified attempts per task. This is 160x the
# 1x4 smoke budget while fitting one 8xA100 node comfortably.
export TTT_RUN_NAME="${TTT_RUN_NAME:-qwen3-4b-ttt-${TASK}-8x16x5}"
export TTT_NUM_ROLLOUT="${TTT_NUM_ROLLOUT:-5}"
export TTT_ROLLOUT_BATCH_SIZE="${TTT_ROLLOUT_BATCH_SIZE:-8}"
export TTT_SAMPLES_PER_PROMPT="${TTT_SAMPLES_PER_PROMPT:-16}"
export TTT_GLOBAL_BATCH_SIZE="${TTT_GLOBAL_BATCH_SIZE:-128}"
export TTT_RESPONSE_LEN="${TTT_RESPONSE_LEN:-8192}"
export TTT_PHASE1_CONTEXT="${TTT_PHASE1_CONTEXT:-4096}"
export TTT_MAX_TOKENS_PER_GPU="${TTT_MAX_TOKENS_PER_GPU:-8192}"
export TTT_EVAL_TIMEOUT="${TTT_EVAL_TIMEOUT:-90}"
export TTT_NUM_CPUS_PER_TASK="${TTT_NUM_CPUS_PER_TASK:-2}"
export TTT_EVAL_CONCURRENCY="${TTT_EVAL_CONCURRENCY:-32}"
export TTT_MAX_BUFFER_SIZE="${TTT_MAX_BUFFER_SIZE:-1000}"
export TTT_SAVE_INTERVAL="${TTT_SAVE_INTERVAL:-5}"

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_ttt_erdos_smoke.sh"

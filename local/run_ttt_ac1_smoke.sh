#!/usr/bin/env bash
set -euo pipefail

export TTT_RUN_NAME="${TTT_RUN_NAME:-qwen3-4b-ttt-ac1-smoke}"
export TTT_ENV_PATH="examples.ttt_discover.envs.ac_inequalities.AC1Env"
export TTT_TARGET="1.5030"
export TTT_NUM_ROLLOUT="${TTT_NUM_ROLLOUT:-1}"
export TTT_ROLLOUT_BATCH_SIZE="${TTT_ROLLOUT_BATCH_SIZE:-1}"
export TTT_SAMPLES_PER_PROMPT="${TTT_SAMPLES_PER_PROMPT:-4}"
export TTT_GLOBAL_BATCH_SIZE="${TTT_GLOBAL_BATCH_SIZE:-4}"
export TTT_EVAL_TIMEOUT="${TTT_EVAL_TIMEOUT:-90}"
export TTT_NUM_CPUS_PER_TASK="${TTT_NUM_CPUS_PER_TASK:-2}"
export TTT_RESPONSE_LEN="${TTT_RESPONSE_LEN:-4096}"
export TTT_PHASE1_CONTEXT="${TTT_PHASE1_CONTEXT:-1536}"

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_ttt_erdos_smoke.sh"

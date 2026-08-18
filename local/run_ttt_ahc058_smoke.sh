#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_DIR}/env.sh"

export TTT_ENV_PATH="examples.ttt_discover.envs.ahc058.AHC058Env"
export TTT_TARGET="${TTT_TARGET:-6500000}"
export TTT_RUN_NAME="${TTT_RUN_NAME:-qwen3-4b-ttt-ahc058-smoke}"
export TTT_AHC_CACHE_DIR="${TTT_AHC_CACHE_DIR}"
export TTT_AHC_REFERENCE_ROOT="${TTT_AHC_REFERENCE_ROOT:-${TTT_OFFICIAL_ROOT}}"
# Five public inputs match ALE-Bench lite mode and keep the first end-to-end run
# cheap. Raise this to 50 for the official public subset or 100 for all released
# generated inputs in the repository.
export TTT_AHC_MAX_CASES="${TTT_AHC_MAX_CASES:-5}"
export TTT_AHC_TIME_LIMIT="${TTT_AHC_TIME_LIMIT:-2}"
export TTT_RESPONSE_LEN="${TTT_RESPONSE_LEN:-8192}"
export TTT_PHASE1_CONTEXT="${TTT_PHASE1_CONTEXT:-8192}"
export TTT_EVAL_TIMEOUT="${TTT_EVAL_TIMEOUT:-120}"
export TTT_EVAL_CONCURRENCY="${TTT_EVAL_CONCURRENCY:-8}"
export TTT_NUM_CPUS_PER_TASK="${TTT_NUM_CPUS_PER_TASK:-2}"
# Official AHC058 starts from an empty program.  The 4B local smoke explicitly
# warm-starts from a 5/5-AC greedy seed so it tests improvement rather than
# one-shot synthesis of a long valid contest solution.
export TTT_AHC058_WARM_START="${TTT_AHC058_WARM_START:-1}"
export TTT_AHC058_SEED="${TTT_AHC058_SEED:-${TTT_ROOT}/local/ahc058_greedy_baseline.cpp}"

exec "${LOCAL_DIR}/run_ttt_erdos_smoke.sh"

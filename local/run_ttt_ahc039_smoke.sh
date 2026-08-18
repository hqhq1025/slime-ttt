#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_DIR}/env.sh"
CACHE=${AHC_CACHE:-${TTT_AHC_CACHE_DIR}}
REFERENCE=${TTT_OFFICIAL_ROOT}

test -x "$CACHE/tester_binaries/ahc039_tester"
test -f "$REFERENCE/examples/ahc/prompt.py"

export TTT_ENV_PATH=examples.ttt_discover.envs.ahc039.AHC039Env
export TTT_TARGET=${TTT_TARGET:-5000}
export TTT_RUN_NAME=${TTT_RUN_NAME:-qwen3-4b-ttt-ahc039-smoke}
export TTT_NUM_ROLLOUT=${TTT_NUM_ROLLOUT:-1}
export TTT_ROLLOUT_BATCH_SIZE=${TTT_ROLLOUT_BATCH_SIZE:-1}
export TTT_SAMPLES_PER_PROMPT=${TTT_SAMPLES_PER_PROMPT:-4}
# The default 8-GPU layout has data-parallel size 4, so Megatron requires the
# global batch size to be divisible by 4.
export TTT_GLOBAL_BATCH_SIZE=${TTT_GLOBAL_BATCH_SIZE:-4}
export TTT_RESPONSE_LEN=${TTT_RESPONSE_LEN:-8192}
# The released AHC039 statement plus seed solution is ~12,955 Qwen3 tokens.
# Full C++ candidates exceeded 4k tokens in the first end-to-end run. Leave
# enough room for an 8k candidate and size training for the full sequence.
export TTT_PHASE1_CONTEXT=${TTT_PHASE1_CONTEXT:-24576}
export TTT_MAX_TOKENS_PER_GPU=${TTT_MAX_TOKENS_PER_GPU:-24576}
export TTT_EVAL_TIMEOUT=${TTT_EVAL_TIMEOUT:-120}
export TTT_EVAL_CONCURRENCY=${TTT_EVAL_CONCURRENCY:-2}
export TTT_MAX_BUFFER_SIZE=${TTT_MAX_BUFFER_SIZE:-16}

# CLI defaults point at these paths. Keep the checks above explicit so a missing
# official artifact fails before reserving GPUs.
export TTT_AHC_CACHE_DIR="$CACHE"
export TTT_AHC_REFERENCE_ROOT="$REFERENCE"
export TTT_AHC_MAX_CASES="${AHC_MAX_CASES:-1}"
exec bash "${LOCAL_DIR}/run_ttt_erdos_smoke.sh"

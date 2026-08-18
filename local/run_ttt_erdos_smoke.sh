#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_DIR}/env.sh"
source "${LOCAL_DIR}/model_qwen3_4b.sh"

HF_MODEL="${TTT_STORAGE_ROOT}/models/Qwen3-4B"
MCORE_MODEL="${TTT_STORAGE_ROOT}/models/Qwen3-4B_torch_dist"
RUN_NAME="${TTT_RUN_NAME:-${TTT_SMOKE_RUN_NAME:-qwen3-4b-ttt-erdos-faithful-smoke}}"
ENV_PATH="${TTT_ENV_PATH:-examples.ttt_discover.envs.erdos.ErdosMinOverlapEnv}"
TARGET="${TTT_TARGET:-0.3808}"
SAVE_DIR="${TTT_STORAGE_ROOT}/checkpoints/${RUN_NAME}"
# Ray appends a long session name and socket suffix; keep this path short enough
# for Linux's 107-byte AF_UNIX pathname limit while remaining on /data.
RAY_DIR="${TTT_RAY_DIR:-${TTT_STORAGE_ROOT}/ray_tmp}"
LOG_DIR="${TTT_STORAGE_ROOT}/logs/${RUN_NAME}"
RESPONSE_LEN="${TTT_RESPONSE_LEN:-${TTT_SMOKE_RESPONSE_LEN:-4096}}"
PHASE1_CONTEXT="${TTT_PHASE1_CONTEXT:-${TTT_SMOKE_PHASE1_CONTEXT:-1536}}"
MAX_TOKENS_PER_GPU="${TTT_MAX_TOKENS_PER_GPU:-${TTT_SMOKE_MAX_TOKENS_PER_GPU:-8192}}"
START_ROLLOUT="${TTT_START_ROLLOUT:-0}"
NUM_ROLLOUT="${TTT_NUM_ROLLOUT:-1}"
ROLLOUT_BATCH_SIZE="${TTT_ROLLOUT_BATCH_SIZE:-1}"
SAMPLES_PER_PROMPT="${TTT_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${TTT_GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * SAMPLES_PER_PROMPT))}"
SAVE_INTERVAL="${TTT_SAVE_INTERVAL:-1}"
EVAL_CONCURRENCY="${TTT_EVAL_CONCURRENCY:-2}"
EVAL_TIMEOUT="${TTT_EVAL_TIMEOUT:-60}"
NUM_CPUS_PER_TASK="${TTT_NUM_CPUS_PER_TASK:-2}"
MAX_BUFFER_SIZE="${TTT_MAX_BUFFER_SIZE:-16}"
TRIMUL_PROFILE="${TTT_TRIMUL_PROFILE:-smoke}"
TRIMUL_REPEATS="${TTT_TRIMUL_REPEATS:-5}"
TRIMUL_GPU_DEVICE="${TTT_TRIMUL_GPU_DEVICE:-0}"
MLA_REPEATS="${TTT_MLA_REPEATS:-3}"
MLA_GPU_DEVICE="${TTT_MLA_GPU_DEVICE:-0}"
MLA_OFFICIAL_ROOT="${TTT_MLA_OFFICIAL_ROOT:-${TTT_OFFICIAL_ROOT}}"
AHC_CACHE_DIR="${TTT_AHC_CACHE_DIR}"
AHC_REFERENCE_ROOT="${TTT_AHC_REFERENCE_ROOT:-${TTT_OFFICIAL_ROOT}}"
AHC_MAX_CASES="${TTT_AHC_MAX_CASES:-1}"
AHC_TIME_LIMIT="${TTT_AHC_TIME_LIMIT:-2.0}"
NUM_GPUS="${NUM_GPUS:-8}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
# The matched 8 groups x 16 candidates run averages 32 sequences per TP=2
# engine. Two-phase asynchronous generation can transiently skew above 32,
# so capture through 64 and retain an override for tighter-memory machines.
SGLANG_CUDA_GRAPH_MAX_BS="${TTT_SGLANG_CUDA_GRAPH_MAX_BS:-64}"
LORA_RANK="${TTT_LORA_RANK:-0}"
LORA_ALPHA="${TTT_LORA_ALPHA:-32}"
LORA_DROPOUT="${TTT_LORA_DROPOUT:-0.0}"
LR="${TTT_LR:-1e-6}"
EXTRA_MODEL_ARGS=()
EXTRA_SGLANG_ARGS=()
if (( LORA_RANK > 0 )); then
  # The base checkpoint predates adapter parameters. Load all matching frozen
  # weights and retain the freshly initialized LoRA A/B matrices.
  EXTRA_MODEL_ARGS+=(--dist-ckpt-strictness ignore_all)
fi
if [[ "${TTT_SGLANG_DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  EXTRA_SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi

[[ -s "${HF_MODEL}/config.json" ]] || { echo "Missing ${HF_MODEL}; run local/prepare_qwen3_4b.sh" >&2; exit 1; }
[[ -s "${MCORE_MODEL}/latest_checkpointed_iteration.txt" ]] || { echo "Missing ${MCORE_MODEL}; run local/prepare_qwen3_4b.sh" >&2; exit 1; }
mkdir -p "${SAVE_DIR}" "${RAY_DIR}" "${LOG_DIR}"

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -oE 'NV[0-9]+' | wc -l)"
if (( NVLINK_COUNT > 0 )); then export NCCL_NVLS_ENABLE=1; else export NCCL_NVLS_ENABLE=0; fi

# Own Ray in a dedicated process group and only terminate that group on exit.
setsid ray start --head --block \
  --node-ip-address=127.0.0.1 \
  --port="${RAY_PORT}" \
  --num-gpus="${NUM_GPUS}" \
  --temp-dir="${RAY_DIR}" \
  --disable-usage-stats \
  --dashboard-host=127.0.0.1 \
  --dashboard-port="${DASHBOARD_PORT}" \
  >"${LOG_DIR}/ray-head.log" 2>&1 &
RAY_GROUP_PID=$!

cleanup() {
  trap - EXIT INT TERM
  if kill -0 "${RAY_GROUP_PID}" 2>/dev/null; then
    kill -TERM -- "-${RAY_GROUP_PID}" 2>/dev/null || true
    wait "${RAY_GROUP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${DASHBOARD_PORT}/api/version" >/dev/null 2>&1; then break; fi
  kill -0 "${RAY_GROUP_PID}" 2>/dev/null || { tail -100 "${LOG_DIR}/ray-head.log"; exit 1; }
  sleep 1
done
curl -fsS "http://127.0.0.1:${DASHBOARD_PORT}/api/version" >/dev/null

cd "${SLIME_PATH}"
RUNTIME_ENV_JSON="$(python - <<PY
import json
print(json.dumps({"env_vars": {
    "PATH": "${PATH}",
    "LD_LIBRARY_PATH": "${LD_LIBRARY_PATH}",
    "PYTHONPATH": "${PYTHONPATH}",
    "HF_ENDPOINT": "${HF_ENDPOINT}",
    "HF_HOME": "${HF_HOME}",
    "HF_HUB_CACHE": "${HF_HUB_CACHE}",
    "TRANSFORMERS_CACHE": "${TRANSFORMERS_CACHE}",
    "TORCH_HOME": "${TORCH_HOME}",
    "XDG_CACHE_HOME": "${XDG_CACHE_HOME}",
    "FLASHINFER_WORKSPACE_BASE": "${FLASHINFER_WORKSPACE_BASE}",
    "TMPDIR": "${TMPDIR}",
    "CUDA_HOME": "${CUDA_HOME}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${NCCL_NVLS_ENABLE}",
}}))
PY
)"

ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python "${SLIME_PATH}/examples/ttt_discover/train_ttt.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${EXTRA_MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_MODEL}" \
  --ref-load "${MCORE_MODEL}" \
  --load "${MCORE_MODEL}" \
  --save "${SAVE_DIR}" \
  --save-interval "${SAVE_INTERVAL}" \
  --rollout-function-path examples.ttt_discover.ttt_rollout.generate_rollout \
  --ttt-env-path "${ENV_PATH}" \
  --ttt-target "${TARGET}" \
  --ttt-lora-rank "${LORA_RANK}" \
  --ttt-lora-alpha "${LORA_ALPHA}" \
  --ttt-lora-dropout "${LORA_DROPOUT}" \
  --ttt-eval-timeout "${EVAL_TIMEOUT}" \
  --ttt-num-cpus-per-task "${NUM_CPUS_PER_TASK}" \
  --ttt-eval-concurrency "${EVAL_CONCURRENCY}" \
  --ttt-max-buffer-size "${MAX_BUFFER_SIZE}" \
  --ttt-trimul-profile "${TRIMUL_PROFILE}" \
  --ttt-trimul-repeats "${TRIMUL_REPEATS}" \
  --ttt-trimul-gpu-device "${TRIMUL_GPU_DEVICE}" \
  --ttt-mla-repeats "${MLA_REPEATS}" \
  --ttt-mla-gpu-device "${MLA_GPU_DEVICE}" \
  --ttt-mla-official-root "${MLA_OFFICIAL_ROOT}" \
  --ttt-ahc-cache-dir "${AHC_CACHE_DIR}" \
  --ttt-ahc-reference-root "${AHC_REFERENCE_ROOT}" \
  --ttt-ahc-max-cases "${AHC_MAX_CASES}" \
  --ttt-ahc-time-limit "${AHC_TIME_LIMIT}" \
  --ttt-topk-children 2 \
  --ttt-phase1-max-context "${PHASE1_CONTEXT}" \
  --ttt-context-window 40960 \
  --disable-rollout-global-dataset \
  --apply-chat-template \
  --start-rollout-id "${START_ROLLOUT}" \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${SAMPLES_PER_PROMPT}" \
  --rollout-max-response-len "${RESPONSE_LEN}" \
  --rollout-temperature 1.0 \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --balance-data \
  --advantage-estimator entropic_adaptive_beta \
  --adv-entropic-target-kl 0.6931 \
  --custom-advantage-function-path examples.ttt_discover.advantage.compute_ttt_advantages \
  --kl-coef 0.1 \
  --use-rollout-logprobs \
  --eps-clip inf \
  --eps-clip-high inf \
  --optimizer adam \
  --lr "${LR}" \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --tensor-model-parallel-size 2 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
  --rollout-num-gpus-per-engine 2 \
  --sglang-mem-fraction-static 0.7 \
  --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}" \
  "${EXTRA_SGLANG_ARGS[@]}" \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash

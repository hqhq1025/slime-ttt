#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_DIR}/env.sh"

CANDIDATE="${1:-${TTT_OFFICIAL_ROOT}/results/kernel-engineering/trimul.py}"
PROFILE="${2:-smoke}"
REPEATS="${TTT_TRIMUL_REPEATS:-5}"
GPU_DEVICE="${TTT_TRIMUL_GPU_DEVICE:-0}"
OUT_DIR="${TTT_STORAGE_ROOT}/logs/trimul-a100-eval"
mkdir -p "${OUT_DIR}"

[[ -s "${CANDIDATE}" ]] || { echo "Missing candidate: ${CANDIDATE}" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
exec python "${SLIME_PATH}/examples/ttt_discover/trimul_eval.py" \
  "${CANDIDATE}" \
  --profile "${PROFILE}" \
  --repeats "${REPEATS}" \
  --output "${OUT_DIR}/result-${PROFILE}.json"

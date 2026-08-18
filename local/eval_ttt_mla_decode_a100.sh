#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_DIR}/env.sh"

CANDIDATE="${1:-${TTT_OFFICIAL_ROOT}/results/kernel-engineering/mla_code_3.py}"
REPEATS="${TTT_MLA_REPEATS:-3}"
SUITE="${TTT_MLA_SUITE:-smoke}"
GPU_DEVICE="${TTT_MLA_GPU_DEVICE:-0}"
OUT_DIR="${TTT_STORAGE_ROOT}/logs/mla-decode-a100-eval"
CANDIDATE_NAME="$(basename -- "${CANDIDATE}" .py)"
mkdir -p "${OUT_DIR}"

# Isolate torch.compile/Triton temporary files from concurrent experiments.
MLA_TMPDIR="${TTT_STORAGE_ROOT}/tmp/mla-decode-${SUITE}-${CANDIDATE_NAME}"
mkdir -p "${MLA_TMPDIR}"
export TMPDIR="${MLA_TMPDIR}"

[[ -s "${CANDIDATE}" ]] || { echo "Missing candidate: ${CANDIDATE}" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
exec python "${SLIME_PATH}/examples/ttt_discover/mla_decode_a100_eval.py" \
  "${CANDIDATE}" \
  --official-root "${TTT_OFFICIAL_ROOT}" \
  --repeats "${REPEATS}" \
  --suite "${SUITE}" \
  --output "${OUT_DIR}/result-${SUITE}-${CANDIDATE_NAME}.json"

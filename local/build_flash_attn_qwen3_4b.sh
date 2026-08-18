#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_DIR}/env.sh"

SOURCE_DIR="${TTT_ROOT}/flash-attn-src"
LOG_FILE="${TTT_ROOT}/logs/flash_attn_build_qwen128.log"
CCCL_INCLUDE="$(find "${CUDA_HOME}/components" -maxdepth 2 -type d -path '*/cuda_cccl-*/include' -print -quit)"

[[ -f "${SOURCE_DIR}/setup.py" ]] || { echo "Missing ${SOURCE_DIR}" >&2; exit 1; }
[[ -n "${CCCL_INCLUDE}" ]] || { echo "CUDA CCCL headers not found below ${CUDA_HOME}" >&2; exit 1; }
mkdir -p "$(dirname -- "${LOG_FILE}")"

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
export FLASH_ATTN_CUDA_ARCHS=80
export FLASH_ATTN_QWEN3_4B_ONLY=1
export MAX_JOBS="${MAX_JOBS:-4}"
# The assembled CUDA toolkit uses a symlink farm. Prefer CCCL's physical
# include directory to avoid transient shared-filesystem symlink lookup errors.
export NVCC_PREPEND_FLAGS="-I${CCCL_INCLUDE} ${NVCC_PREPEND_FLAGS:-}"

cd "${SOURCE_DIR}"
python -m pip install -v --no-build-isolation --no-deps --force-reinstall . \
  2>&1 | tee -a "${LOG_FILE}"

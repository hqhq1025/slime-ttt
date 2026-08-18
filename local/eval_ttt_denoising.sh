#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_DIR}/env.sh"
CANDIDATE="${1:-${TTT_OFFICIAL_ROOT}/results/denoising/denoise_ttt.py}"

export TMPDIR="${TTT_STORAGE_ROOT}/tmp/denoising"
export OPENPROBLEMS_CACHE_DIR="${OPENPROBLEMS_CACHE_DIR:-${TTT_STORAGE_ROOT}/openproblems_cache}"
export OPENPROBLEMS_PANCREAS_PATH="${OPENPROBLEMS_PANCREAS_PATH:-${TTT_STORAGE_ROOT}/datasets/openproblems/pancreas.h5ad}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TTT_STORAGE_ROOT}/numba_cache}"
export PYTHONPATH="${TTT_OFFICIAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${TMPDIR}" "${OPENPROBLEMS_CACHE_DIR}" "${NUMBA_CACHE_DIR}"
exec "${TTT_DENOISING_PYTHON:-${TTT_STORAGE_ROOT}/denoising-venv/bin/python}" \
  "${LOCAL_DIR}/evaluate_denoising.py" "${CANDIDATE}"

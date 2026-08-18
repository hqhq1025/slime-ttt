#!/usr/bin/env bash

# Shared native environment for the local TTT-Discover reproduction. Every
# location is overrideable so the checkout can move between clusters.
TTT_LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TTT_ROOT="${TTT_ROOT:-$(cd -- "${TTT_LOCAL_DIR}/.." && pwd)}"
export TTT_STORAGE_ROOT="${TTT_STORAGE_ROOT:-$(dirname -- "${TTT_ROOT}")/ttt-storage}"
export TTT_VENV="${TTT_VENV:-${TTT_ROOT}/.venv}"
export CUDA_HOME="${CUDA_HOME:-${TTT_ROOT}/cuda-12.8}"
export MEGATRON_PATH="${MEGATRON_PATH:-${TTT_ROOT}/Megatron-LM}"
export SLIME_PATH="${SLIME_PATH:-${TTT_ROOT}/slime}"
export SGLANG_SOURCE="${SGLANG_SOURCE:-${TTT_ROOT}/sglang/python}"
export TTT_OFFICIAL_ROOT="${TTT_OFFICIAL_ROOT:-$(dirname -- "${TTT_ROOT}")/ttt-discover-official}"
export TTT_AHC_CACHE_DIR="${TTT_AHC_CACHE_DIR:-${TTT_STORAGE_ROOT}/ahc-cache/extracted/cache}"

export PATH="${TTT_VENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# slime and Megatron both ship a top-level `examples` package. Keep slime
# first so examples.ttt_discover resolves to the local TTT implementation.
export PYTHONPATH="${SLIME_PATH}:${MEGATRON_PATH}:${SGLANG_SOURCE}:${PYTHONPATH:-}"

# PyTorch's CUDA component wheels keep development headers/libraries under
# site-packages/nvidia rather than CUDA_HOME. Expose them to native builds.
TTT_PYTHON_VERSION="$(${TTT_VENV}/bin/python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
TTT_NVIDIA_ROOT="${TTT_VENV}/lib/${TTT_PYTHON_VERSION:-python3.12}/site-packages/nvidia"
for TTT_COMPONENT_DIR in "${TTT_NVIDIA_ROOT}"/*; do
  [[ -d "${TTT_COMPONENT_DIR}/include" ]] && CPLUS_INCLUDE_PATH="${TTT_COMPONENT_DIR}/include:${CPLUS_INCLUDE_PATH:-}"
  [[ -d "${TTT_COMPONENT_DIR}/lib" ]] && LD_LIBRARY_PATH="${TTT_COMPONENT_DIR}/lib:${LD_LIBRARY_PATH}"
done
export CPLUS_INCLUDE_PATH LD_LIBRARY_PATH
unset TTT_COMPONENT_DIR

# Keep every large/reproducible artifact on /data.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${TTT_STORAGE_ROOT}/hf_cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${TORCH_HOME:-${TTT_STORAGE_ROOT}/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TTT_STORAGE_ROOT}/cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${TTT_STORAGE_ROOT}}"
# Keep build temp outside the Git worktree: setuptools-scm otherwise mistakes
# the slime-ttt commit for the version of third-party source distributions.
export TMPDIR="${TMPDIR:-${TTT_STORAGE_ROOT}/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-${TTT_STORAGE_ROOT}/ray_tmp}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Shared /data storage does not preserve uv's cache hardlinks reliably.
export UV_LINK_MODE=copy

mkdir -p "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" "${TMPDIR}" "${RAY_TMPDIR}" \
  "${TTT_STORAGE_ROOT}/logs" "${TTT_STORAGE_ROOT}/checkpoints" "${TTT_STORAGE_ROOT}/models"

unset TTT_LOCAL_DIR TTT_PYTHON_VERSION

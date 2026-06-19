#!/usr/bin/env bash
# Full-parameter TTT-Discover: gpt-oss-20B, Frontier-CS, 4xH200, single node (g3125).
# Everything (container, caches, ckpts, logs) stays under /gscratch/zlab/lky04/slime-ttt.
# Usage:
#   PROBLEM_ID=0 JUDGE_BACKEND=local bash scripts/run-ttt-frontiercs-gptoss20b.sh
#   JUDGE_BACKEND=remote JUDGE_URL=http://<judge-host>/score bash scripts/run-ttt-frontiercs-gptoss20b.sh
set -uo pipefail
PROJ=/gscratch/zlab/lky04/slime-ttt
SIF=$PROJ/apptainer/images/slime.sif
export PROBLEM_ID=${PROBLEM_ID:-0}
export JUDGE_BACKEND=${JUDGE_BACKEND:-remote}
export JUDGE_URL=${JUDGE_URL:-https://yanagiorigami.uk}
export NUM_GPUS=${NUM_GPUS:-4}

echo "[run] $(date -Is) on $(hostname); problem=$PROBLEM_ID judge=$JUDGE_BACKEND"
nvidia-smi -L || true

exec apptainer exec --nv --no-home \
  --bind /gscratch:/gscratch \
  --bind $PROJ/slime:/root/slime \
  --bind $PROJ/cache:/cache --bind $PROJ/hf_cache:/hf_cache \
  --env HF_HOME=/hf_cache --env HUGGINGFACE_HUB_CACHE=/hf_cache --env XDG_CACHE_HOME=/cache \
  --env TRITON_CACHE_DIR=/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/cache/inductor --env TORCH_HOME=/cache/torch \
  --env TOKENIZERS_PARALLELISM=false --env CUDA_DEVICE_MAX_CONNECTIONS=1 --env SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  --env PYTHONPATH=/root/slime:/root/Megatron-LM \
  --env PROJ=$PROJ --env NUM_GPUS=$NUM_GPUS --env PROBLEM_ID=$PROBLEM_ID \
  --env JUDGE_BACKEND=$JUDGE_BACKEND --env JUDGE_URL=$JUDGE_URL \
  "$SIF" bash $PROJ/scripts/train_inner.sh

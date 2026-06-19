#!/usr/bin/env bash
# Download Qwen/Qwen3.6-35B-A3B and convert to a Megatron torch_dist checkpoint.
# Megatron model type = qwen3.5-35B-A3B (Qwen3.6 shares the qwen3.5 arch; per slime's
# own tests/test_qwen3.6_35B_A3B_pd_mooncake.py). Everything under zlab/lky04. Idempotent.
set -uo pipefail
PROJ=/gscratch/zlab/lky04/slime-ttt
SIF=${SIF:-$PROJ/apptainer/images/slime-latest.sif}   # Qwen3.6 hybrid checkpoint needs latest megatron-core
M=$PROJ/models
MODEL=Qwen3.6-35B-A3B
MTYPE=qwen3.5-35B-A3B
mkdir -p "$M" "$PROJ/logs"

BINDS=(--no-home --bind /gscratch:/gscratch
  --bind "$PROJ/cache:/cache" --bind "$PROJ/hf_cache:/hf_cache"
  --env HF_HOME=/hf_cache --env HUGGINGFACE_HUB_CACHE=/hf_cache --env XDG_CACHE_HOME=/cache
  --env TRITON_CACHE_DIR=/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/cache/inductor
  --env TORCH_HOME=/cache/torch --env TOKENIZERS_PARALLELISM=false
  --env PYTHONPATH=/root/slime:/root/Megatron-LM
  --env CUDA_DEVICE_MAX_CONNECTIONS=1 --env HF_HUB_ENABLE_HF_TRANSFER=0)

echo "[prep-qwen] $(date -Is) on $(hostname)"; nvidia-smi -L || true

# 1) download HF checkpoint (already bf16, no dequant needed) -> models/Qwen3.6-35B-A3B
if [ ! -f "$M/$MODEL/config.json" ]; then
  echo "[prep-qwen] downloading Qwen/$MODEL ..."
  apptainer exec "${BINDS[@]}" "$SIF" hf download "Qwen/$MODEL" --local-dir "$M/$MODEL" \
    || { echo "[prep-qwen] download FAILED"; exit 1; }
else echo "[prep-qwen] $MODEL already present, skip download"; fi

# 2) convert HF bf16 -> Megatron torch_dist (4 GPUs on this node, via AutoBridge/qwen3_5)
if [ ! -d "$M/${MODEL}_torch_dist" ] || [ -z "$(ls -A "$M/${MODEL}_torch_dist" 2>/dev/null)" ]; then
  echo "[prep-qwen] convert -> torch_dist ..."
  apptainer exec --nv "${BINDS[@]}" "$SIF" bash -lc \
    "cd /root/slime && source scripts/models/${MTYPE}.sh && \
     PYTHONPATH=/root/slime:/root/Megatron-LM torchrun --nproc_per_node 4 \
     tools/convert_hf_to_torch_dist.py \
     --hf-checkpoint $M/$MODEL --save $M/${MODEL}_torch_dist \
     \${MODEL_ARGS[@]}" \
    || { echo "[prep-qwen] convert FAILED"; exit 1; }
else echo "[prep-qwen] torch_dist already present, skip convert"; fi

echo "[prep-qwen] DONE $(date -Is)"; du -sh "$M"/${MODEL}* 2>/dev/null

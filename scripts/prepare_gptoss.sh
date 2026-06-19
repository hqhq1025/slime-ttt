#!/usr/bin/env bash
# Download + convert openai/gpt-oss-20b to a Megatron torch_dist checkpoint for slime.
# Everything stays under /gscratch/zlab/lky04/slime-ttt. Idempotent per step.
set -uo pipefail
PROJ=/gscratch/zlab/lky04/slime-ttt
SIF=$PROJ/apptainer/images/slime.sif
M=$PROJ/models
mkdir -p "$M"

BINDS=(--no-home --bind /gscratch:/gscratch --bind "$PROJ/slime:/root/slime"
  --bind "$PROJ/cache:/cache" --bind "$PROJ/hf_cache:/hf_cache"
  --env HF_HOME=/hf_cache --env HUGGINGFACE_HUB_CACHE=/hf_cache --env XDG_CACHE_HOME=/cache
  --env TRITON_CACHE_DIR=/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/cache/inductor
  --env TORCH_HOME=/cache/torch --env TOKENIZERS_PARALLELISM=false
  --env CUDA_DEVICE_MAX_CONNECTIONS=1 --env HF_HUB_ENABLE_HF_TRANSFER=0)

echo "[prep] $(date -Is) on $(hostname)"; nvidia-smi -L || true

# 1) download HF checkpoint (MXFP4) -> models/gpt-oss-20b
if [ ! -f "$M/gpt-oss-20b/config.json" ]; then
  echo "[prep] downloading openai/gpt-oss-20b ..."
  apptainer exec "${BINDS[@]}" "$SIF" hf download openai/gpt-oss-20b --local-dir "$M/gpt-oss-20b" || { echo "[prep] download FAILED"; exit 1; }
else echo "[prep] gpt-oss-20b already present, skip download"; fi

# 2) dequantize MXFP4 -> bf16
if [ ! -f "$M/gpt-oss-20b-bf16/config.json" ]; then
  echo "[prep] preprocess -> bf16 ..."
  apptainer exec --nv "${BINDS[@]}" "$SIF" bash -lc "cd /root/slime && python tools/preprocess_gpt_oss.py --input $M/gpt-oss-20b --output $M/gpt-oss-20b-bf16" || { echo "[prep] preprocess FAILED"; exit 1; }
else echo "[prep] bf16 already present, skip preprocess"; fi

# 3) convert bf16 -> Megatron torch_dist (4 GPUs)
if [ ! -d "$M/gpt-oss-20b_torch_dist" ] || [ -z "$(ls -A $M/gpt-oss-20b_torch_dist 2>/dev/null)" ]; then
  echo "[prep] convert -> torch_dist ..."
  apptainer exec --nv "${BINDS[@]}" "$SIF" bash -lc "cd /root/slime && source scripts/models/gpt-oss-20B.sh && PYTHONPATH=/root/slime:/root/Megatron-LM torchrun --nproc_per_node 4 tools/convert_hf_to_torch_dist.py --hf-checkpoint $M/gpt-oss-20b-bf16 --save $M/gpt-oss-20b_torch_dist --megatron-to-hf-mode bridge \${MODEL_ARGS[@]}" || { echo "[prep] convert FAILED"; exit 1; }
else echo "[prep] torch_dist already present, skip convert"; fi

echo "[prep] DONE $(date -Is)"; du -sh "$M"/* 2>/dev/null

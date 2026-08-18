#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_DIR}/env.sh"
source "${LOCAL_DIR}/model_qwen3_4b.sh"

HF_MODEL="${TTT_STORAGE_ROOT}/models/Qwen3-4B"
MCORE_MODEL="${TTT_STORAGE_ROOT}/models/Qwen3-4B_torch_dist"

HF_MODEL_READY=1
for required_file in \
  config.json tokenizer.json tokenizer_config.json model.safetensors.index.json \
  model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
  [[ -s "${HF_MODEL}/${required_file}" ]] || HF_MODEL_READY=0
done
if (( HF_MODEL_READY )) && ! python - "${HF_MODEL}/tokenizer_config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    config = json.load(f)
assert config.get("chat_template"), "tokenizer_config.json has no chat_template"
PY
then
  HF_MODEL_READY=0
fi

if (( ! HF_MODEL_READY )); then
  hf download Qwen/Qwen3-4B --local-dir "${HF_MODEL}" --max-workers 8
fi

python - "${HF_MODEL}/tokenizer_config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    config = json.load(f)
assert config.get("chat_template"), "Qwen3 tokenizer must provide a chat_template"
PY

if [[ ! -s "${MCORE_MODEL}/latest_checkpointed_iteration.txt" ]]; then
  cd "${SLIME_PATH}"
  python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_MODEL}" \
    --save "${MCORE_MODEL}"
fi

echo "HF checkpoint:       ${HF_MODEL}"
echo "Megatron checkpoint: ${MCORE_MODEL}"

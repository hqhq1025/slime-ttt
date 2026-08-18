#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?usage: run_ttt_domain_lora_scale.sh ac1|ac2|circle26|circle32}"
LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TTT_RUN_NAME="${TTT_RUN_NAME:-qwen3-4b-ttt-${TASK}-lora-r32-8x16x5}"
export TTT_LORA_RANK="${TTT_LORA_RANK:-32}"
export TTT_LORA_ALPHA="${TTT_LORA_ALPHA:-32}"
export TTT_LORA_DROPOUT="${TTT_LORA_DROPOUT:-0.0}"
export TTT_LR="${TTT_LR:-4e-5}"
exec "${LOCAL_DIR}/run_ttt_domain_scale.sh" "${TASK}"

#!/usr/bin/env bash
set -uo pipefail

LOCAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${LOCAL_DIR}/env.sh"

RESULTS="${TTT_LORA_SUITE_RESULTS:-${TTT_STORAGE_ROOT}/logs/lora-cross-domain-suite.tsv}"
mkdir -p "$(dirname -- "${RESULTS}")"
if [[ "${TTT_LORA_SUITE_APPEND:-0}" != "1" || ! -s "${RESULTS}" ]]; then
  printf 'task\tmode\tstatus\tstarted\tfinished\n' >"${RESULTS}"
fi

START_AT="${TTT_LORA_SUITE_START_AT:-ac1}"
ACTIVE=0
stage_enabled() {
  local stage=$1
  if (( ACTIVE == 0 )) && [[ "${stage}" == "${START_AT}" ]]; then
    ACTIVE=1
  fi
  (( ACTIVE == 1 ))
}

run_one() {
  local task=$1 mode=$2
  shift 2
  local started finished status
  started=$(date -u +%FT%TZ)
  echo "[LoRA suite] START ${task} ${mode} ${started}"
  if "$@"; then status=ok; else status=failed; fi
  finished=$(date -u +%FT%TZ)
  printf '%s\t%s\t%s\t%s\t%s\n' "${task}" "${mode}" "${status}" "${started}" "${finished}" >>"${RESULTS}"
  echo "[LoRA suite] END ${task} ${mode} ${status} ${finished}"
}

# Exact matched-budget comparisons against completed full-parameter runs.
for task in ac1 ac2 circle26 circle32; do
  if stage_enabled "${task}"; then
    run_one "${task}" lora "${LOCAL_DIR}/run_ttt_domain_lora_scale.sh" "${task}"
  fi
done
if stage_enabled trimul; then
  run_one trimul lora "${LOCAL_DIR}/run_ttt_trimul_lora_scale.sh"
fi

# Fresh two-step pairs: step 0 trains and step 1 measures the adapted policy.
if stage_enabled mla; then
  run_one mla full env \
    TTT_RUN_NAME=qwen3-4b-ttt-mla-a100-full-pair-2step \
    TTT_NUM_ROLLOUT=2 \
    "${LOCAL_DIR}/run_ttt_mla_decode_a100_smoke.sh"
  run_one mla lora env \
    TTT_RUN_NAME=qwen3-4b-ttt-mla-a100-lora-r32-pair-2step \
    TTT_NUM_ROLLOUT=2 TTT_LORA_RANK=32 TTT_LORA_ALPHA=32 TTT_LR=4e-5 \
    "${LOCAL_DIR}/run_ttt_mla_decode_a100_smoke.sh"
fi

for task in ahc039 ahc058; do
  if stage_enabled "${task}"; then
    launcher="${LOCAL_DIR}/run_ttt_${task}_smoke.sh"
    run_one "${task}" full env \
      TTT_RUN_NAME=qwen3-4b-ttt-${task}-full-pair-2step \
      TTT_NUM_ROLLOUT=2 \
      "${launcher}"
    run_one "${task}" lora env \
      TTT_RUN_NAME=qwen3-4b-ttt-${task}-lora-r32-pair-2step \
      TTT_NUM_ROLLOUT=2 TTT_LORA_RANK=32 TTT_LORA_ALPHA=32 TTT_LR=4e-5 \
      "${launcher}"
  fi
done

if stage_enabled denoising; then
  run_one denoising full env \
    TTT_RUN_NAME=qwen3-4b-ttt-denoising-full-pair-2step \
    TTT_NUM_ROLLOUT=2 \
    "${LOCAL_DIR}/run_ttt_denoising_smoke.sh"
  run_one denoising lora env \
    TTT_RUN_NAME=qwen3-4b-ttt-denoising-lora-r32-pair-2step \
    TTT_NUM_ROLLOUT=2 TTT_LORA_RANK=32 TTT_LORA_ALPHA=32 TTT_LR=4e-5 \
    "${LOCAL_DIR}/run_ttt_denoising_smoke.sh"
fi

echo "[LoRA suite] results: ${RESULTS}"

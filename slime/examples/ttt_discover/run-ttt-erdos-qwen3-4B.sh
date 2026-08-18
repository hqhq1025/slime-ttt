#!/bin/bash
# ============================================================================
# Full-parameter TTT-Discover on slime — Erdős minimum-overlap, Qwen3-4B, 1 node.
#
# This runs the "Learning to Discover at Test Time" (arXiv:2601.16175) algorithm
# on slime's native Megatron (FULL-PARAMETER, no LoRA) + SGLang stack:
#   - custom rollout = the discovery loop (archive + sandbox reward)
#   - advantage      = entropic_adaptive_beta (the paper's discovery objective)
#   - KL-to-base     = --use-kl-loss against --ref-load
#
# Prereqs (same as scripts/run-qwen3-4B.sh):
#   - Qwen3-4B HF checkpoint at /root/Qwen3-4B
#   - torch_dist checkpoint at /root/Qwen3-4B_torch_dist (also used as ref)
#   - Megatron-LM importable (PYTHONPATH below)
# Security: generated code is executed in a sandbox subprocess. Run isolated.
# ============================================================================

pkill -9 sglang; sleep 3; ray stop --force; pkill -9 ray; pkill -9 python; sleep 3
pkill -9 ray; pkill -9 python

set -ex
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0
echo "HAS_NVLINK: $HAS_NVLINK"

DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS:-8}}
[ "$NUM_GPUS" -le 0 ] && NUM_GPUS=8
echo "NUM_GPUS: $NUM_GPUS"

# Resolve the slime repo root from this script's location (examples/ttt_discover/).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"   # defines MODEL_ARGS

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   --ref-load /root/Qwen3-4B_torch_dist          # base model for the KL penalty
   --load /root/Qwen3-4B_slime_ttt/
   --save /root/Qwen3-4B_slime_ttt/
   --save-interval 10
)

# ---- TTT discovery loop ----------------------------------------------------
# No --prompt-data: prompts are generated from the discovery archive, so the
# global dataset is disabled and the custom rollout drives everything.
TTT_ARGS=(
   --rollout-function-path examples.ttt_discover.ttt_rollout.generate_rollout
   --ttt-env-path examples.ttt_discover.envs.erdos.ErdosMinOverlapEnv
   --ttt-target 0.3808
   --ttt-eval-timeout 120
   --ttt-num-cpus-per-task 2
   --ttt-eval-concurrency 32
   --ttt-max-buffer-size 1000
   --ttt-topk-children 2
   --ttt-phase1-max-context 12288 # reserve response budget for final code
   --ttt-context-window 32768
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --apply-chat-template
   --num-rollout 50                 # ~ TTT-Discover num_epochs (single-problem)
   --rollout-batch-size 8           # groups per step  (TTT groups_per_batch)
   --n-samples-per-prompt 16        # group size       (TTT group_size=64 on bigger HW)
   --rollout-max-response-len 16384 # reasoning + code (paper uses ~26k for gpt-oss)
   --rollout-temperature 1.0
   --global-batch-size 128          # = rollout-batch-size * n-samples-per-prompt
   --balance-data
)

# ---- the TTT objective -----------------------------------------------------
ALGO_ARGS=(
   --advantage-estimator entropic_adaptive_beta   # << the discovery objective
   --adv-entropic-target-kl 0.6931                # log(2), the paper's default
   # Official TTT: KL is part of the token advantage, not a separate KL loss.
   --custom-advantage-function-path examples.ttt_discover.advantage.compute_ttt_advantages
   --kl-coef 0.1
   # Official importance-sampling PG uses sampler log-probs without PPO clipping.
   --use-rollout-logprobs
   --eps-clip inf
   --eps-clip-high inf
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6                        # full-param LR (paper's 4e-5 was for LoRA)
   --lr-decay-style constant
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.95
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 18432
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=(
   # --use-wandb --wandb-project slime-ttt --wandb-group erdos-qwen3-4B
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# NOTE: entry point is the TTT wrapper (registers --ttt-* args), not train.py.
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 examples/ttt_discover/train_ttt.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${TTT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${ALGO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]}

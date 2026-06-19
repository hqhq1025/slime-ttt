#!/usr/bin/env bash
# Runs INSIDE the slime.sif container. Launches ray + full-parameter TTT training.
set -ex
cd /root/slime
source scripts/models/gpt-oss-20B.sh    # defines MODEL_ARGS

PROJ=${PROJ:-/gscratch/zlab/lky04/slime-ttt}
M=$PROJ/models
CKPT=$PROJ/ckpts
NUM_GPUS=${NUM_GPUS:-4}
PROBLEM_ID=${PROBLEM_ID:-0}
JUDGE_BACKEND=${JUDGE_BACKEND:-local}
JUDGE_URL=${JUDGE_URL:-}
mkdir -p "$CKPT" "$PROJ/logs"

pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 2
ray start --head --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host 0.0.0.0 --dashboard-port 8265

JUDGE_ARG=""
if [ -n "$JUDGE_URL" ]; then JUDGE_ARG="--ttt-judge-url $JUDGE_URL"; fi
STAMP=$(date +%Y%m%d_%H%M%S)

python train_ttt.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node ${NUM_GPUS} --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint ${M}/gpt-oss-20b-bf16 \
  --ref-load ${M}/gpt-oss-20b_torch_dist \
  --load ${CKPT} --save ${CKPT} --save-interval 10 \
  --rollout-function-path examples.ttt_discover.ttt_rollout.generate_rollout \
  --ttt-env-path examples.ttt_discover.envs.frontiercs.FrontierCSEnv \
  --ttt-frontiercs-problems-dir ${PROJ}/frontiercs/problems \
  --ttt-frontiercs-problem-id ${PROBLEM_ID} \
  --ttt-frontiercs-max-score 100.0 \
  --ttt-judge-backend ${JUDGE_BACKEND} ${JUDGE_ARG} \
  --ttt-eval-timeout 120 --ttt-eval-concurrency 16 --ttt-judge-max-cases 1 \
  --disable-rollout-global-dataset \
  --apply-chat-template \
  --num-rollout 50 \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 24576 \
  --rollout-temperature 1.0 \
  --global-batch-size 32 \
  --advantage-estimator entropic_adaptive_beta \
  --adv-entropic-target-kl 0.6931 \
  --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.01 \
  --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0 --micro-batch-size 1 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 4 \
  --expert-tensor-parallel-size 1 \
  --use-distributed-optimizer \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 8192 \
  --moe-token-dispatcher-type alltoall \
  --megatron-to-hf-mode bridge \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --rollout-num-gpus ${NUM_GPUS} \
  --sglang-mem-fraction-static 0.5 \
  2>&1 | tee ${PROJ}/logs/train_${STAMP}.log

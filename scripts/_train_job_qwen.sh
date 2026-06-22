#!/usr/bin/env bash
# Runs INSIDE the head container; submits the Qwen3.6-35B-A3B TTT training to the ray cluster.
# Qwen3.6 shares the qwen3.5-35B-A3B Megatron arch (slime's own qwen3.6 test maps it so).
# Parallelism mirrors that test scaled 8->16 GPUs: TP2 . CP2 . PP1 . EP8 . DP4 (16 GPU world),
# flash attention (no learnable-softmax restriction unlike gpt-oss), recompute + cpu-offload
# optimizer for the 35B full-parameter state.
set -e
PROJ=/gscratch/zlab/lky04/slime-ttt
cd /root/slime
source scripts/models/qwen3.5-35B-A3B.sh   # MODEL_ARGS (qwen3_5 spec, 40L/256E, vocab 248320, etc.)

PROBLEM_ID=${PROBLEM_ID:-159}
JUDGE_BACKEND=${JUDGE_BACKEND:-remote}
JUDGE_URL=${JUDGE_URL:-https://yanagiorigami.uk}
RBS=${RBS:-4}; NSAMP=${NSAMP:-16}; GBS=$((RBS*NSAMP))
MODEL=Qwen3.6-35B-A3B
CKPT=$PROJ/ckpts/qwen-p${PROBLEM_ID}            # per-problem: fresh from --ref-load unless a real ckpt exists
mkdir -p "$CKPT" "$PROJ/logs"
LOAD_ARG=""; [ -f "$CKPT/latest_checkpointed_iteration.txt" ] && LOAD_ARG="--load $CKPT"
WPROJ=${WPROJ:-ttt-frontiercs-qwen36-35ba3b}; WGROUP=qwen36-p${PROBLEM_ID}-${RBS}x${NSAMP}
JUDGE_ARG=""; [ -n "$JUDGE_URL" ] && JUDGE_ARG="--ttt-judge-url $JUDGE_URL"
echo "[job] QWEN3.6-35B-A3B problem=$PROBLEM_ID groups=${RBS}x${NSAMP}=${GBS} ckpt=$CKPT load=[$LOAD_ARG] wandb=$WPROJ/$WGROUP"

ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"/root/slime:/root/Megatron-LM\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"SGLANG_ENABLE_JIT_DEEPGEMM\":\"0\",\"TOKENIZERS_PARALLELISM\":\"false\",\"WANDB_API_KEY\":\"${WANDB_API_KEY:-}\",\"WANDB_DIR\":\"$PROJ/logs/wandb\",\"WANDB_CACHE_DIR\":\"$PROJ/cache/wandb\",\"WANDB_CONFIG_DIR\":\"$PROJ/cache/wandb\"}}" \
  -- python3 /root/slime/examples/ttt_discover/train_ttt.py \
  --actor-num-nodes 4 --actor-num-gpus-per-node 4 --num-gpus-per-node 4 --colocate \
  "${MODEL_ARGS[@]}" \
  --seq-length 30720 \
  --hf-checkpoint $PROJ/models/$MODEL \
  --ref-load $PROJ/models/${MODEL}_torch_dist \
  $LOAD_ARG --save $CKPT --save-interval 5 \
  --rollout-function-path examples.ttt_discover.ttt_rollout.generate_rollout \
  --ttt-env-path examples.ttt_discover.envs.frontiercs.FrontierCSEnv \
  --ttt-frontiercs-problems-dir $PROJ/frontiercs/problems \
  --ttt-frontiercs-problem-id $PROBLEM_ID --ttt-frontiercs-max-score 100.0 \
  --ttt-judge-backend $JUDGE_BACKEND $JUDGE_ARG \
  --ttt-eval-timeout 300 --ttt-eval-concurrency 32 --ttt-judge-max-cases 1 \
  --disable-rollout-global-dataset --apply-chat-template \
  --num-rollout 50 --rollout-batch-size $RBS --n-samples-per-prompt $NSAMP \
  --rollout-max-response-len 26000 --rollout-temperature 1.0 --global-batch-size $GBS \
  --advantage-estimator entropic_adaptive_beta --adv-entropic-target-kl 0.6931 \
  --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.01 \
  --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0 --micro-batch-size 1 \
  --tensor-model-parallel-size 2 --sequence-parallel \
  --pipeline-model-parallel-size 1 --context-parallel-size 2 \
  --expert-model-parallel-size 8 --expert-tensor-parallel-size 1 --use-distributed-optimizer \
  --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash \
  --rollout-num-gpus 16 --rollout-num-gpus-per-engine 2 \
  --sglang-mem-fraction-static 0.6 --sglang-max-running-requests 64 \
  --sglang-disable-custom-all-reduce \
  --sglang-router-request-timeout-secs 3600 \
  --use-wandb --wandb-project $WPROJ --wandb-group $WGROUP

#!/usr/bin/env bash
# Runs INSIDE the head container; submits the TTT training job to the running ray cluster.
set -e
PROJ=/gscratch/zlab/lky04/slime-ttt
cd /root/slime
source scripts/models/gpt-oss-20B.sh   # MODEL_ARGS (its --seq-length 4096 overridden below)

PROBLEM_ID=${PROBLEM_ID:-0}
JUDGE_BACKEND=${JUDGE_BACKEND:-remote}
JUDGE_URL=${JUDGE_URL:-https://yanagiorigami.uk}
RBS=${RBS:-4}; NSAMP=${NSAMP:-16}; GBS=$((RBS*NSAMP))
CKPT=$PROJ/ckpts/p${PROBLEM_ID}                 # per-problem: each problem trains fresh from --ref-load
mkdir -p "$CKPT" "$PROJ/logs"
LOAD_ARG=""; [ -f "$CKPT/latest_checkpointed_iteration.txt" ] && LOAD_ARG="--load $CKPT"
WPROJ=${WPROJ:-ttt-frontiercs-gptoss20b}; WGROUP=gptoss20b-p${PROBLEM_ID}-${RBS}x${NSAMP}
JUDGE_ARG=""; [ -n "$JUDGE_URL" ] && JUDGE_ARG="--ttt-judge-url $JUDGE_URL"
echo "[job] problem=$PROBLEM_ID groups=${RBS}x${NSAMP}=${GBS} ckpt=$CKPT load=[$LOAD_ARG] wandb=$WPROJ/$WGROUP"

ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"/root/slime:/root/Megatron-LM\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"SGLANG_ENABLE_JIT_DEEPGEMM\":\"0\",\"TOKENIZERS_PARALLELISM\":\"false\",\"WANDB_API_KEY\":\"${WANDB_API_KEY:-}\",\"WANDB_DIR\":\"$PROJ/logs/wandb\",\"WANDB_CACHE_DIR\":\"$PROJ/cache/wandb\",\"WANDB_CONFIG_DIR\":\"$PROJ/cache/wandb\"}}" \
  -- python3 /root/slime/examples/ttt_discover/train_ttt.py \
  --actor-num-nodes 4 --actor-num-gpus-per-node 4 --num-gpus-per-node 4 --colocate \
  "${MODEL_ARGS[@]}" \
  --seq-length 30720 \
  --hf-checkpoint $PROJ/models/gpt-oss-20b-bf16 \
  --ref-load $PROJ/models/gpt-oss-20b_torch_dist \
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
  --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0 --micro-batch-size 1 --qkv-format bshd \
  --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 --use-distributed-optimizer \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --moe-token-dispatcher-type alltoall --megatron-to-hf-mode bridge \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend fused \
  --rollout-num-gpus 16 --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.5 --sglang-cuda-graph-max-bs 16 --sglang-max-running-requests 64 \
  --use-wandb --wandb-project $WPROJ --wandb-group $WGROUP

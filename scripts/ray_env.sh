# Shared env for the containerized multi-node Ray cluster. SLURM-aware:
# under sbatch it derives nodes from $SLURM_JOB_NODELIST; else falls back to the
# interactive node list. Sourced on the HOST (head node) by ray_up/submit/monitor.
PROJ=/gscratch/zlab/lky04/slime-ttt
SIF=${SIF:-$PROJ/apptainer/images/slime.sif}
GPUS_PER_NODE=4
PORT=6379
DASH=8265
RAYTMP=/tmp/rayttt_lky04
TMPFLAG="--temp-dir $RAYTMP"
RAYMEM=460000000000          # 460 GB/node (respect 512G/node SLURM grant)
RAYOBJ=40000000000           # 40 GB/node plasma
MEMFLAGS="--memory $RAYMEM --object-store-memory $RAYOBJ"

if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
  _NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
else
  _NODES=(g3125 g3127 g3128 g3130)     # interactive fallback (update if reallocated)
fi
HEAD_HOST=${_NODES[0]}
HEAD_IP=$(getent ahostsv4 "$HEAD_HOST" | awk "{print \$1}" | grep -m1 "^10[.]")
WORKER_IPS=""
for _n in "${_NODES[@]:1}"; do WORKER_IPS="$WORKER_IPS $(getent ahostsv4 "$_n" | awk "{print \$1}" | grep -m1 "^10[.]")"; done
WORKER_IPS=$(echo $WORKER_IPS | xargs)
ALL_IPS="$HEAD_IP $WORKER_IPS"
NUM_NODES=${#_NODES[@]}

source $PROJ/scripts/wandb_secret.sh 2>/dev/null || true

BINDS="--nv --no-home --bind /gscratch:/gscratch --bind $PROJ/slime:/root/slime --bind $PROJ/cache:/cache --bind $PROJ/hf_cache:/hf_cache"
ENVS="--env HF_HOME=/hf_cache --env HUGGINGFACE_HUB_CACHE=/hf_cache --env XDG_CACHE_HOME=/cache \
--env TRITON_CACHE_DIR=/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/cache/inductor --env TORCH_HOME=/cache/torch \
--env TOKENIZERS_PARALLELISM=false --env SGLANG_ENABLE_JIT_DEEPGEMM=0 \
--env APPTAINER_CACHEDIR=$PROJ/cache/apptainer --env APPTAINERTMPDIR=$PROJ/cache/apptainer_tmp \
--env CUDA_DEVICE_MAX_CONNECTIONS=1 --env PYTHONPATH=/root/slime:/root/Megatron-LM \
--env NCCL_SOCKET_IFNAME=ens11f0np0 --env GLOO_SOCKET_IFNAME=ens11f0np0 --env NCCL_DEBUG=WARN \
--env RAY_ADDRESS=${HEAD_IP}:${PORT} \
--env WANDB_DIR=$PROJ/logs/wandb --env WANDB_CACHE_DIR=$PROJ/cache/wandb --env WANDB_CONFIG_DIR=$PROJ/cache/wandb \
--env WANDB_API_KEY=${WANDB_API_KEY:-}"

#!/bin/bash
#SBATCH --job-name=ttt-gptoss20b-frontiercs
#SBATCH --account=zlab
#SBATCH --partition=gpu-h200
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/zlab/lky04/slime-ttt/logs/sbatch_%j.out
#SBATCH --error=/gscratch/zlab/lky04/slime-ttt/logs/sbatch_%j.out
# Full-parameter TTT-Discover: gpt-oss-20B, Frontier-CS, 4 nodes x 4 H200.
# Submit:  cd /gscratch/zlab/lky04/slime-ttt && sbatch scripts/submit.sbatch
# Override: PROBLEM_ID=3 RBS=4 NSAMP=16 sbatch --export=ALL scripts/submit.sbatch
set -uo pipefail
P=/gscratch/zlab/lky04/slime-ttt
export PROBLEM_ID=${PROBLEM_ID:-0}
export RBS=${RBS:-4}
export NSAMP=${NSAMP:-16}
export JUDGE_BACKEND=${JUDGE_BACKEND:-remote}
export JUDGE_URL=${JUDGE_URL:-https://yanagiorigami.uk}
echo "[sbatch] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST problem=$PROBLEM_ID grp=${RBS}x${NSAMP} start=$(date -Is)"
nvidia-smi -L || true
trap "echo [sbatch] cleanup; bash $P/scripts/ray_down.sh || true" EXIT
bash $P/scripts/ray_up.sh          # ray head+workers across the SLURM nodes (capped, pinned NIC)
bash $P/scripts/submit_train.sh    # run training to completion (ray job submit streams; blocks)
echo "[sbatch] done $(date -Is)"

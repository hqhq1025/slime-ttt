#!/bin/bash
#SBATCH --job-name=ttt-qwen36-frontiercs
#SBATCH --account=zlab
#SBATCH --partition=gpu-h200
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/zlab/lky04/slime-ttt/logs/sbatch_qwen_%j.out
#SBATCH --error=/gscratch/zlab/lky04/slime-ttt/logs/sbatch_qwen_%j.out
# Full-parameter TTT-Discover: Qwen3.6-35B-A3B, Frontier-CS, 4 nodes x 4 H200.
# Submit:  cd /gscratch/zlab/lky04/slime-ttt && sbatch scripts/submit_qwen.sbatch
# Override: PROBLEM_ID=159 RBS=4 NSAMP=16 sbatch --export=ALL scripts/submit_qwen.sbatch
set -uo pipefail
P=/gscratch/zlab/lky04/slime-ttt
export SIF=$P/apptainer/images/slime-latest.sif   # Qwen3.6 image (ray_up + train both inherit this)
export PROBLEM_ID=${PROBLEM_ID:-159}
export RBS=${RBS:-4}
export NSAMP=${NSAMP:-16}
export JUDGE_BACKEND=${JUDGE_BACKEND:-remote}
export JUDGE_URL=${JUDGE_URL:-https://yanagiorigami.uk}
echo "[sbatch] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST problem=$PROBLEM_ID grp=${RBS}x${NSAMP} start=$(date -Is)"
nvidia-smi -L || true
trap "echo [sbatch] cleanup; bash $P/scripts/ray_down.sh || true" EXIT
bash $P/scripts/ray_up.sh                # ray head+workers across the SLURM nodes (capped, pinned NIC)
bash $P/scripts/submit_train_qwen.sh     # run Qwen3.6 training to completion (blocks)
echo "[sbatch] done $(date -Is)"

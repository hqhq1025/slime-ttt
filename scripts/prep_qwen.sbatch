#!/bin/bash
#SBATCH --job-name=prep-qwen36
#SBATCH --account=zlab
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=6:00:00
#SBATCH --output=/gscratch/zlab/lky04/slime-ttt/logs/prep_qwen_%j.out
#SBATCH --error=/gscratch/zlab/lky04/slime-ttt/logs/prep_qwen_%j.out
# One-shot: download + convert Qwen3.6-35B-A3B on a single 4xH200 node.
# Submit:  cd /gscratch/zlab/lky04/slime-ttt && sbatch scripts/prep_qwen.sbatch
set -uo pipefail
P=/gscratch/zlab/lky04/slime-ttt
echo "[prep.sbatch] job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST start=$(date -Is)"
nvidia-smi -L || true
bash "$P/scripts/prepare_qwen.sh"
echo "[prep.sbatch] done $(date -Is)"

#!/bin/bash
#SBATCH --job-name=pull-slime-latest
#SBATCH --account=zlab
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/zlab/lky04/slime-ttt/logs/pull_latest_%j.out
#SBATCH --error=/gscratch/zlab/lky04/slime-ttt/logs/pull_latest_%j.out
# Convert slimerl/slime:latest -> slime-latest.sif on a compute node (latest megatron-core
# matches the qwen3.5/3.6 plugin; v0.3.0's was too old for the hybrid checkpoint).
set -uo pipefail
P=/gscratch/zlab/lky04/slime-ttt
echo "[pull-latest] job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST start=$(date -Is)"
export APPTAINER_CACHEDIR="$P/cache/apptainer"
export APPTAINER_TMPDIR="$P/cache/apt_tmp"
mkdir -p "$APPTAINER_TMPDIR"
rm -rf "$APPTAINER_TMPDIR"/build-temp-* "$APPTAINER_TMPDIR"/bundle-temp-* 2>/dev/null || true
cd "$P/apptainer/images"
echo "[pull-latest] converting docker://slimerl/slime:latest -> slime-latest.sif ..."
apptainer pull --force slime-latest.sif docker://slimerl/slime:latest
echo "[pull-latest] apptainer exit=$?"
ls -la "$P/apptainer/images/slime-latest.sif" 2>/dev/null || echo "NO SIF PRODUCED"
echo "[pull-latest] done $(date -Is)"

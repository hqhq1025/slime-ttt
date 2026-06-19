#!/bin/bash
#SBATCH --job-name=pull-slime-v030
#SBATCH --account=zlab
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/zlab/lky04/slime-ttt/logs/pull_v030_%j.out
#SBATCH --error=/gscratch/zlab/lky04/slime-ttt/logs/pull_v030_%j.out
# Convert slimerl/slime:v0.3.0 (already-cached blobs) -> slime-v030.sif on a compute node.
# Login node killed the OCI->SIF conversion; a compute node has the resources.
set -uo pipefail
P=/gscratch/zlab/lky04/slime-ttt
echo "[pull.sbatch] job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST start=$(date -Is)"
export APPTAINER_CACHEDIR="$P/cache/apptainer"          # reuse the 23G of blobs already downloaded
export APPTAINER_TMPDIR="$P/cache/apt_tmp"              # gpfs: huge, no tmpfs/OOM risk
mkdir -p "$APPTAINER_TMPDIR"
rm -rf "$APPTAINER_TMPDIR"/build-temp-* "$APPTAINER_TMPDIR"/bundle-temp-* 2>/dev/null || true
cd "$P/apptainer/images"
echo "[pull.sbatch] converting docker://slimerl/slime:v0.3.0 -> slime-v030.sif ..."
apptainer pull --force slime-v030.sif docker://slimerl/slime:v0.3.0
RC=$?
echo "[pull.sbatch] apptainer exit=$RC"
ls -la "$P/apptainer/images/slime-v030.sif" 2>/dev/null || echo "NO SIF PRODUCED"
echo "[pull.sbatch] done $(date -Is)"

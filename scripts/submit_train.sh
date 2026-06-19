#!/usr/bin/env bash
set -uo pipefail
source /gscratch/zlab/lky04/slime-ttt/scripts/ray_env.sh
apptainer exec $BINDS $ENVS \
  --env PROBLEM_ID=${PROBLEM_ID:-0} --env JUDGE_URL=${JUDGE_URL:-https://yanagiorigami.uk} \
  --env RBS=${RBS:-4} --env NSAMP=${NSAMP:-16} \
  $SIF bash /gscratch/zlab/lky04/slime-ttt/scripts/_train_job.sh

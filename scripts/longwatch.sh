#!/usr/bin/env bash
source /gscratch/zlab/lky04/slime-ttt/scripts/ray_env.sh
JID="$1"
for i in $(seq 1 50); do
  ST=$(apptainer exec $BINDS $ENVS $SIF ray job status $JID 2>/dev/null | grep -oE "RUNNING|FAILED|STOPPED|SUCCEEDED" | tail -1)
  if [ "$ST" != "RUNNING" ]; then
    echo "JOB ENDED: $ST @iter $i"
    apptainer exec $BINDS $ENVS $SIF ray job logs $JID 2>/dev/null | grep -oE "\[TTT\] step [0-9]+: .*archive=[0-9]+|train/step.: [0-9]+|Error|out of memory" | tail -8
    exit 0
  fi
  sleep 180
done
echo "STILL RUNNING after long watch"
apptainer exec $BINDS $ENVS $SIF ray job logs $JID 2>/dev/null | grep -oE "\[TTT\] step [0-9]+: .*best_raw=[0-9.]+.*archive=[0-9]+" | tail -5

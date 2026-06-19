#!/usr/bin/env bash
source /gscratch/zlab/lky04/slime-ttt/scripts/ray_env.sh
JID="$1"; [ -z "$JID" ] && JID=$(grep -oE "raysubmit_[A-Za-z0-9]+" /gscratch/zlab/lky04/slime-ttt/logs/train_submit.log | tail -1)
echo "monitoring $JID"
for i in $(seq 1 60); do
  ST=$(apptainer exec $BINDS $ENVS $SIF ray job status $JID 2>/dev/null | grep -oE "RUNNING|FAILED|STOPPED|SUCCEEDED" | tail -1)
  LOG=$(apptainer exec $BINDS $ENVS $SIF ray job logs $JID 2>/dev/null | grep -ivE "socket.send|ServerArgs|update_weights_from_tensor|health_generate|Gloo. Rank 0" | tail -600)
  if echo "$LOG" | grep -qE "\[TTT\] step [1-9]"; then echo "VERDICT: FULL LOOP OK (train step completed) @$i"; echo "$LOG" | grep -E "\[TTT\] step|Timer train end|grad" | tail -8; exit 0; fi
  if echo "$LOG" | grep -qiE "No dot product attention|NVTE"; then echo "VERDICT: TE ATTENTION ERROR @$i"; echo "$LOG" | grep -iE "No dot product attention|NVTE|backend|disabl" | tail -10; exit 4; fi
  if echo "$LOG" | grep -qiE "out of memory|OutOfMemoryError|CUDA error|illegal memory"; then echo "VERDICT: OOM @$i"; echo "$LOG" | grep -iE "out of memory|cuda error" | tail -5; exit 2; fi
  if echo "$LOG" | grep -qiE "ActorDiedError|RuntimeError|ValueError|AssertionError" && echo "$LOG" | grep -qE "Timer train end|forward_step"; then echo "VERDICT: TRAIN-STEP ERROR @$i"; echo "$LOG" | grep -iE "Error|exception" | tail -8; exit 5; fi
  if [ "$ST" = "FAILED" ] || [ "$ST" = "STOPPED" ]; then echo "VERDICT: JOB $ST @$i"; echo "$LOG" | grep -iE "Error|\[TTT\]|Timer train" | tail -12; exit 3; fi
  sleep 45
done
echo "VERDICT: TIMEOUT"; echo "$LOG" | grep -iE "\[TTT\]|Timer|Error" | tail -10

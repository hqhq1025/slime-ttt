#!/usr/bin/env bash
set -uo pipefail
source /gscratch/zlab/lky04/slime-ttt/scripts/ray_env.sh
bash $PROJ/scripts/ray_down.sh; sleep 4
echo "[ray_up] head on $HEAD_IP"
nohup apptainer exec $BINDS $ENVS $SIF ray start --head --node-ip-address $HEAD_IP --port $PORT $TMPFLAG $MEMFLAGS --num-gpus $GPUS_PER_NODE --dashboard-host 0.0.0.0 --dashboard-port $DASH --disable-usage-stats --block > $PROJ/logs/ray_head.log 2>&1 &
for i in $(seq 1 40); do grep -q "Ray runtime started" $PROJ/logs/ray_head.log 2>/dev/null && break; sleep 2; done
for ip in $WORKER_IPS; do
  echo "[ray_up] worker on $ip"
  ssh -o StrictHostKeyChecking=no $ip "nohup apptainer exec $BINDS $ENVS $SIF ray start --address=${HEAD_IP}:${PORT} --node-ip-address ${ip} $TMPFLAG $MEMFLAGS --num-gpus $GPUS_PER_NODE --disable-usage-stats --block > $PROJ/logs/ray_worker_${ip}.log 2>&1 < /dev/null &" </dev/null
done
echo "[ray_up] waiting for 16 GPUs..."
for i in $(seq 1 40); do
  L=$(apptainer exec $BINDS $ENVS $SIF ray status 2>/dev/null | grep -E "/[0-9.]+ GPU")
  echo "  [$i] $L"
  echo "$L" | grep -q "/16.0 GPU" && break
  sleep 5
done
echo "[ray_up] === final ray status ==="
apptainer exec $BINDS $ENVS $SIF ray status 2>&1 | head -25

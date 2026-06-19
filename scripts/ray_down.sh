#!/usr/bin/env bash
source /gscratch/zlab/lky04/slime-ttt/scripts/ray_env.sh
for ip in $ALL_IPS; do
  ssh -o StrictHostKeyChecking=no $ip "ray stop --force 2>/dev/null; pkill -9 -f raylet; pkill -9 -f gcs_server; pkill -9 -f plasma_store; pkill -9 sglang; rm -rf $RAYTMP 2>/dev/null; true" </dev/null
done
echo "[ray_down] done"

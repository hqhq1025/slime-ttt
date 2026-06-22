#!/usr/bin/env bash
# Run ON the head compute node: summarize the live Qwen3.6 TTT run from Ray's logs.
D=$(ls -dt /tmp/rayttt_lky04/session_*/logs 2>/dev/null | head -1)
J="$D"/job-driver-*.log
echo "STEP=$(grep -ho 'TTT] step [0-9]*' $J 2>/dev/null | tail -1 | grep -o '[0-9]*$')"
echo "CIRC=$(cat $J 2>/dev/null | grep -c no_available_workers)"
echo "ERR=$(cat "$D"/worker-*.err 2>/dev/null | grep -cE 'Scheduler hit an exception|out of memory|CUDA error|AssertionError|illegal memory|RuntimeError')"
echo "--- recent TTT / score lines ---"
grep -hE 'TTT] step|raw_score|best_raw|archive=' $J 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -8
echo "--- latest engine decode (throughput / cuda graph) ---"
grep -hE 'Decode batch' "$D"/worker-*.err 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -1

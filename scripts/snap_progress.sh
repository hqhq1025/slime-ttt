#!/usr/bin/env bash
# Run ON the head compute node: snapshot the live [TTT] step progression from Ray's
# driver log into a persistent zlab file (survives the end-of-job /tmp cleanup).
D=$(ls -dt /tmp/rayttt_lky04/session_*/logs 2>/dev/null | head -1)
OUT=/gscratch/zlab/lky04/slime-ttt/logs/ttt_progress_latest.txt
grep -ahoE '\[TTT\] step [0-9]+: [0-9]+/[0-9]+ valid, best_raw=[0-9.eE+-]+, archive=[0-9]+' \
  "$D"/job-driver-*.log 2>/dev/null > "$OUT"
echo "snapshotted $(wc -l < "$OUT") steps -> $OUT"

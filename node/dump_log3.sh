#!/bin/sh
# dump_log3.sh — node1: 轮询 chain.log 直到出现首个 [train] step, 然后回传尾部
W=/mnt/workspace/node1/chain
ECS=http://47.99.131.12:8081
i=0
while [ $i -lt 45 ]; do
  if grep -q "step 10/" "$W/chain.log" 2>/dev/null; then break; fi
  i=$((i+1)); sleep 10
done
sleep 20
{ echo "=== waited ${i}0s ==="; tail -c 2200 "$W/chain.log"; echo; echo "=== procs ==="; ps -eo pid,etime,cmd | grep -E 'train_remote_v4|chain_v1' | grep -v grep; } \
  | curl -s -X POST "$ECS/upload?name=chainlog5.txt&key=k0u4mail" --data-binary @-
echo DUMPED

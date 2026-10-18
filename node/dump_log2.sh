#!/bin/sh
# dump_log2.sh — node1: 轮询 chain.log 直到 step 10/ 或超时(30min), 回传尾部 + done_steps + 进程
W=/mnt/workspace/node1/chain
ECS=http://47.99.131.12:8081
i=0
while [ $i -lt 180 ]; do
  if grep -q 'step 10/' "$W/chain.log" 2>/dev/null; then break; fi
  sleep 10; i=$((i+1))
done
{ echo "=== chain.log tail ==="; tail -c 3000 "$W/chain.log"; echo "=== done_steps ==="; cat "$W/done_steps.txt" 2>/dev/null; echo; echo "=== procs ==="; ps aux | grep -E 'train_remote_v4|chain_v1' | grep -v grep; } | curl -s -X POST "$ECS/upload?name=chainlog3.txt&key=k0u4mail" --data-binary @-
echo DUMP2_DONE

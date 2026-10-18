#!/bin/bash
# Upload pristine 200M base shards to ECS /file/ with retry + size verification.
cd "D:/work/_local_ai/train_15b/node" || exit 1
ECS="http://47.99.131.12:8081"
LOG="upload_base.log"
echo "START $(date)" > "$LOG"
for f in parts/r4base_bin.*; do
  n=$(basename "$f")
  local_sz=$(wc -c < "$f")
  ok=0
  for attempt in 1 2 3 4; do
    curl -s -m600 -X POST "$ECS/upload?name=$n&key=k0u4mail" --data-binary @"$f" >/dev/null 2>&1
    sz=$(curl -s -m60 -o /dev/null -w "%{size_download}" "$ECS/file/$n")
    if [ "$sz" = "$local_sz" ]; then
      echo "OK $n ($sz) attempt=$attempt" | tee -a "$LOG"
      ok=1
      break
    fi
    echo "RETRY $n attempt=$attempt got=$sz want=$local_sz" | tee -a "$LOG"
    sleep 3
  done
  if [ "$ok" = 0 ]; then echo "FAIL $n" | tee -a "$LOG"; fi
done
echo "ALL_DONE $(date)" >> "$LOG"

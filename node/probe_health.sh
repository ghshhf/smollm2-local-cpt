#!/bin/bash
# probe_health.sh — node1 侧心跳回显。被本地 health_probe.py 定时派活触发。
# 用法: bash probe_health.sh <unix_ts>  (ts 用于回传文件名去重, 防重名锁)
TS="${1:-$(date +%s)}"
OUT="/tmp/probe_${TS}.txt"
{
  echo "TOK=${TS}"
  echo "alive"
  date "+%H:%M:%S"
  echo "done_steps=$(cat /mnt/workspace/node1/chain/done_steps.txt 2>/dev/null)"
  echo "stage=$(tail -1 /mnt/workspace/node1/chain/heartbeat 2>/dev/null | sed 's/.*stage=//' | head -c 20)"
  echo "train_proc=$(pgrep -c -f train_remote 2>/dev/null || echo 0)"
  uptime
} > "$OUT" 2>&1
curl -s -m 30 --data-binary "@${OUT}" "http://47.99.131.12:8081/upload?name=probe_${TS}.txt&key=k0u4mail" >/dev/null 2>&1
rm -f "$OUT"

#!/usr/bin/env bash
# node1 常驻指令守望器（watchd）
# 协议：轮询 ECS 上的 cmd_000001.txt / cmd_000002.txt / ... 递增序列号；
#       发现新指令就后台执行（绝不阻塞轮询），输出存 cmd_<n>.out 并回传 ECS。
# 幂等：seq 记录在 /mnt/workspace/node1/.cmd_seq，重启/重贴不重复执行。
ECS="http://47.99.131.12:8081"
UP="k0u4mail"
DIR="/mnt/workspace/node1"
SEQF="$DIR/.cmd_seq"
LOG="$DIR/watchd.log"
mkdir -p "$DIR"
[ -f "$SEQF" ] || echo 0 > "$SEQF"

echo "[watchd] started $(date) pid=$$" >> "$LOG"
while true; do
  LAST=$(cat "$SEQF" 2>/dev/null || echo 0)
  for n in $(seq $((LAST+1)) $((LAST+8))); do
    NAME="cmd_$(printf '%06d' $n).txt"
    F=$(curl -s --max-time 15 "$ECS/file/$NAME?key=$UP" 2>/dev/null)
    if [ -n "$F" ] && ! printf '%s' "$F" | head -c 20 | grep -q '"error"'; then
      echo "$n" > "$SEQF"
      echo "[watchd] exec $NAME @ $(date)" >> "$LOG"
      OUT="$DIR/cmd_$(printf '%06d' $n).out"
      nohup bash -c "$F" > "$OUT" 2>&1 &
      # 异步回传执行回执（不阻塞）
      ( sleep 3; curl -s --data-binary @"$OUT" "$ECS/upload?name=watchd_out_$(printf '%06d' $n).txt&key=$UP" >/dev/null 2>&1 ) &
    fi
  done
  sleep 20
done

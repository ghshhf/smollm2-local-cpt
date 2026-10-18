#!/usr/bin/env bash
# maild.sh — node1 常驻信标/信箱守护（协议升级版）
# 机制: 每 60s 把「主链进度 + 侧实验进度 + 负载 + 最新回执」投递为
#       ECS 有序信箱 mail_000001.txt / mail_000002.txt ...（递增序号，永不重名、无锁冲突）
#       本地侧用 node/ecs_mail.py 按序读取，不再盲轮询 /file/<未知名>。
set -u
ECS="http://47.99.131.12:8081"
UP="k0u4mail"
DIR="/mnt/workspace/node1"
SEQF="$DIR/.mail_seq"
mkdir -p "$DIR"
[ -f "$SEQF" ] || echo 0 > "$SEQF"

while true; do
  N=$(( $(cat "$SEQF" 2>/dev/null || echo 0) + 1 ))
  NAME="mail_$(printf '%06d' $N).txt"
  {
    echo "=== beacon $N $(date '+%m-%d %H:%M:%S') load=$(cat /proc/loadavg | cut -d' ' -f1-3) ==="
    echo "--- 主链 chain ---"
    echo "done_steps=$(cat "$DIR/chain/done_steps.txt" 2>/dev/null || echo NA)"
    cat "$DIR/chain/heartbeat" 2>/dev/null | tail -n 1
    ps -eo pid,etime,cmd 2>/dev/null | grep -E "chain_v[0-9]+\.sh|chain_watch|train_remote" | grep -v grep
    echo "--- 侧实验 ---"
    ps -eo pid,etime,cmd 2>/dev/null | grep -E "alpha_ladder|subliminal|run_exps" | grep -v grep
    echo "--- 实验日志 ---"
    tail -n 3 "$DIR/exp_v2.log" 2>/dev/null
    tail -n 3 "$DIR/gh_repo/eval/v1_02_run.log" 2>/dev/null
    tail -n 3 "$DIR/gh_repo/eval/e1_run.log" 2>/dev/null
    echo "--- 结果文件 ---"
    ls -la "$DIR/gh_repo/eval/v1_02_alpha_ladder.json" "$DIR/gh_repo/eval/e1_result.json" 2>/dev/null
  } > /tmp/beacon.txt 2>&1
  curl -s --data-binary @/tmp/beacon.txt "$ECS/upload?name=$NAME&key=$UP" >/dev/null 2>&1
  echo "$N" > "$SEQF"
  sleep 60
done

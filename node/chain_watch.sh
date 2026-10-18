#!/bin/sh
# chain_watch.sh — node1 训练链看门狗(长跑保障)
#
# 为什么需要: node1 上没有任何守护者 —— 链的父进程是 PID 1, node_v5.py 只负责取任务执行,
#             链一旦崩了/被杀就没人拉起来, 训练静默停摆直到人工发现。
#
# 行为: 每 5 分钟检查一次; 没有任何 chain_v*.sh 存活 且 没有 stop 标记 且 课程未跑完 -> 拉起 chain_v1.sh。
#
# 安全性(关键): 即便本脚本误判(比如进程刚被杀但还没消失), 被拉起的新链自身的 guard() 会扫到老链
#               并排队等待接管, 绝不会形成双链 —— 所以"宁可多拉一次"是安全的设计。
#               文件名特意叫 chain_watch.sh: 不匹配 chain_v[0-9]+\.sh, 不会把自己数成一条链。
W=/mnt/workspace/node1/chain
cd "$W" || exit 1

while :; do
  N=$(ps -eo pid,ppid,cmd 2>/dev/null | awk '($3=="sh"||$3=="dash"||$3=="bash"||$3=="/bin/sh") \
      && $4 ~ /chain_v[0-9]+\.sh$/ {n++} END {print n+0}')
  D=$(cat "$W/done_steps.txt" 2>/dev/null || echo 0)
  if [ "$N" -eq 0 ]; then
    if [ -f "$W/nowatch" ]; then
      echo "[$(date '+%m-%d %H:%M:%S')] watchdog: 无链存活, 但存在 nowatch(看门狗已停用), 不拉起" >> "$W/watchdog.log"
    elif [ -f "$W/stop" ]; then
      echo "[$(date '+%m-%d %H:%M:%S')] watchdog: 无链存活, 但检测到 stop 标记(人工暂停/迁移中), 不拉起" >> "$W/watchdog.log"
    elif [ "$D" -ge 12800 ]; then
      echo "[$(date '+%m-%d %H:%M:%S')] watchdog: 无链存活, 但课程已跑完(done=$D >= 12800), 不拉起" >> "$W/watchdog.log"
    else
      echo "[$(date '+%m-%d %H:%M:%S')] watchdog: 无链存活(done=$D) -> 拉起 chain_v1.sh" >> "$W/watchdog.log"
      (nohup sh "$W/chain_v1.sh" > /dev/null 2>&1 < /dev/null &)
    fi
  fi
  sleep 300
done

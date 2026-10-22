#!/bin/bash
# tunnel.sh — node1 → ECS 反向隧道看门狗
# 作用: 把 ECS 的 127.0.0.1:9022 映射回 node1 的 22，本机用 ProxyJump 即可直连 SSH。
# 2026-10-19: 原循环每 5 秒失败一次，因为 ~/.ssh/ecs_tunnel_key 不存在。已修。
LOG=/mnt/workspace/node1/logs/tunnel.log
mkdir -p "$(dirname "$LOG")"
echo "$(date '+%F %T') tunnel watchdog start pid=$$" >> "$LOG"
while true; do
  ssh -N -i /root/.ssh/n1t2 \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes \
      -o ConnectTimeout=10 \
      -R 127.0.0.1:9022:127.0.0.1:22 \
      Administrator@47.99.131.12 >> "$LOG" 2>&1
  rc=$?
  echo "$(date '+%F %T') tunnel down rc=$rc, retry in 5s" >> "$LOG"
  sleep 5
done

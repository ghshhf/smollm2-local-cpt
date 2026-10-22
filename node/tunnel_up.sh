#!/bin/bash
# tunnel_up.sh — 保证 node1 -> ECS 的反向端口隧道常驻
#
# 原理: node1 在 k8s 内网, 没有入向公网; ECS 有公网 22。
#       node1 主动 ssh 到 ECS, 把自己的 22 端口绑到 ECS 的 127.0.0.1:9022。
#       本机再 `ssh -J Administrator@ECS root@127.0.0.1:9022` 就是直连 node1。
#       用处: 多行命令、真实 stdout/exit code、scp、端口转发 (信箱只会单行轮询)。
#
# 用法: bash tunnel_up.sh          # 幂等, 已在跑就跳过
#       bash tunnel_up.sh --status
set -u
LOG=/mnt/workspace/node1/logs/tunnel.log
mkdir -p /mnt/workspace/node1/logs

if [ "${1:-}" = "--status" ]; then
    if pgrep -f 'R 127.0.0.1:9022' >/dev/null; then
        echo "tunnel UP pid=$(pgrep -f 'R 127.0.0.1:9022' | head -1)"
        ss -tlnp 2>/dev/null | grep 9022 || echo "  (本地无 9022, 正常: 监听在 ECS 侧)"
    else
        echo "tunnel DOWN"
    fi
    exit 0
fi

if pgrep -f 'R 127.0.0.1:9022' >/dev/null; then
    echo "tunnel already up"
    exit 0
fi

nohup bash -c 'while true; do
  ssh -N -i ~/.ssh/ecs_tunnel_key \
      -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes \
      -R 127.0.0.1:9022:127.0.0.1:22 Administrator@47.99.131.12
  echo "restart-$(date +%H:%M:%S)"
  sleep 5
done' >> "$LOG" 2>&1 &

sleep 6
if pgrep -f 'R 127.0.0.1:9022' >/dev/null; then
    echo "tunnel up pid=$(pgrep -f 'R 127.0.0.1:9022' | head -1)"
else
    echo "tunnel FAILED, see $LOG"
    tail -5 "$LOG"
    exit 1
fi

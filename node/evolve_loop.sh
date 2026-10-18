#!/bin/bash
# evolve_loop.sh — node1 自进化闭环 · 常驻触发器
# 每 CHECK 分钟跑一次 evolve_cycle.sh（脚本自带闸门：银行新增 < THRESH 直接空转）。
# 由 nohup 常驻启动；上锁防重叠由 cycle 内部 mkdir 锁保证。
set -u
D=/mnt/workspace/node1/dream
cd "$D" || exit 1
CHECK=20
echo "[evolve_loop] $(date '+%F %T') 启动 每${CHECK}min 自检" >> logs/evolve_loop.log
while true; do
  echo "[evolve_loop] $(date '+%F %T') 触发周期" >> logs/evolve_loop.log
  bash evolve_cycle.sh >> logs/evolve_loop.log 2>&1
  sleep $((CHECK * 60))
done

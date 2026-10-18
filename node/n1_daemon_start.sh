#!/bin/bash
# n1_daemon_start.sh — 按参数重启 node1 做梦集群（先拉最新 daemon 脚本）
# 用法: bash n1_daemon_start.sh bare|lora
set -u
D=/mnt/workspace/node1/dream
cd "$D" || exit 1
B="http://47.99.131.12:8081/file"
K="?key=k0u4mail"
MODE="${1:-bare}"
R=/tmp/daemon_start_report.txt
: > "$R"

echo "[daemon] mode=$MODE $(date '+%F %T')" >> "$R"

# 1) 拉最新 daemon 脚本（含 DREAM_LORA 开关）
curl -s -o dream_daemon.sh "$B/dream_daemon.sh$K"
chmod +x dream_daemon.sh
grep -q 'DREAM_LORA' dream_daemon.sh && echo "daemon_ver=new(支持DREAM_LORA)" >> "$R" || echo "daemon_ver=OLD(无DREAM_LORA!)" >> "$R"

# 2) 清场
pkill -f '[d]ream_daemon.sh' 2>/dev/null
sleep 1
pkill -f '[d]ream_loop.py' 2>/dev/null
sleep 3
echo "remaining=$(ps -eo args | grep -cE '[d]ream_loop|[d]ream_daemon')" >> "$R"

# 3) 按模式起集群
if [ "$MODE" = "lora" ]; then
  export DREAM_LORA=/mnt/workspace/node1/models/code_lora_qwen
else
  unset DREAM_LORA 2>/dev/null || true
fi
export DREAM_WORKERS=8
nohup bash dream_daemon.sh > logs/daemon.log 2>&1 &
echo "launched pid=$!" >> "$R"

# 4) 自检：worker 起来没 + 银行快照
sleep 45
echo "workers_now=$(ps -eo args | grep -c '[d]ream_loop.py')" >> "$R"
echo "daemon_head: $(head -1 logs/daemon.log)" >> "$R"
python3 -c "import sqlite3;c=sqlite3.connect('data/triple_bank.db',timeout=30);c.execute('PRAGMA busy_timeout=30000');print('triples=',c.execute('select count(*) from triples').fetchone()[0],'rejects=',c.execute('select count(*) from rejects').fetchone()[0])" >> "$R"
echo "[daemon] DONE $(date '+%F %T')" >> "$R"
cat "$R"

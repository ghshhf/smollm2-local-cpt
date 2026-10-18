#!/bin/bash
# n1_swarm_ctl.sh — node1 自进化集群控制（拉满 + 自适应 + 诊断）
set -u
D=/mnt/workspace/node1/dream
cd "$D" || exit 1
B="http://47.99.131.12:8081/file"
K="?key=k0u4mail"
R=/tmp/swarm_report.txt
: > "$R"

echo "[ctl] === 诊断：其他重负载进程 ===" >> "$R"
ps -eo pid,rss,etimes,args | grep -E '[t]rain_remote_v4|[e]1_subliminal|[v]1_02_alpha|[c]orpus' >> "$R" || echo "(无其他重负载进程)" >> "$R"

echo "[ctl] === 内存 ===" >> "$R"
free -m | head -2 >> "$R"
AVAIL=$(free -m | awk '/^内存|Mem:/{print $7}')
echo "AVAIL_MB=$AVAIL" >> "$R"

echo "[ctl] === 杀旧 dream 进程 ===" >> "$R"
pkill -f '[d]ream_daemon.sh' 2>/dev/null
pkill -f '[d]ream_loop.py' 2>/dev/null
sleep 3
ps -eo pid,args | grep -E '[d]ream_loop|[d]ream_daemon' >> "$R" || echo "old_killed=yes" >> "$R"

echo "[ctl] === 从 ECS 拉三份文件 ===" >> "$R"
curl -s -o tools/triple_bank.py "$B/triple_bank.py$K"; echo "tb=$(wc -c < tools/triple_bank.py)" >> "$R"
curl -s -o tools/dream_elite.py "$B/dream_elite.py$K"; echo "elite=$(wc -c < tools/dream_elite.py)" >> "$R"
curl -s -o dream_daemon.sh      "$B/dream_daemon.sh$K"; echo "daemon=$(wc -c < dream_daemon.sh)" >> "$R"
chmod +x dream_daemon.sh
grep -q journal_mode=WAL tools/triple_bank.py && echo "TB_WAL=1" >> "$R" || echo "TB_WAL=0" >> "$R"
grep -q journal_mode=WAL tools/dream_elite.py && echo "ELITE_WAL=1" >> "$R" || echo "ELITE_WAL=0" >> "$R"

echo "[ctl] === 自适应 worker 数 ===" >> "$R"
if [ "$AVAIL" -lt 16000 ]; then
  WORKERS=2
  echo "内存偏紧(<16G可用)，保守 WORKERS=2" >> "$R"
else
  WORKERS=8
  echo "内存充裕(>=16G可用)，拉满 WORKERS=8" >> "$R"
fi
export DREAM_WORKERS=$WORKERS

echo "[ctl] === 起集群 ===" >> "$R"
nohup bash dream_daemon.sh > logs/daemon.log 2>&1 &
sleep 30
echo "workers_now=$(ps -eo args | grep -c '[d]ream_loop.py')" >> "$R"
echo "daemon_now=$(ps -eo args | grep -c '[d]ream_daemon.sh')" >> "$R"
echo "[ctl] === 启动后内存 ===" >> "$R"; free -m | head -2 >> "$R"
echo "[ctl] === 银行 ===" >> "$R"
python3 -c "import sqlite3;c=sqlite3.connect('data/triple_bank.db',timeout=30);c.execute('PRAGMA busy_timeout=30000');print('triples=',c.execute('select count(*) from triples').fetchone()[0],'rejects=',c.execute('select count(*) from rejects').fetchone()[0])" >> "$R"
echo "[ctl] === daemon.log 尾 ===" >> "$R"; tail -6 logs/daemon.log >> "$R"
echo "[ctl] DONE" >> "$R"
cat "$R"

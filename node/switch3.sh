W=$(ls -d /mnt/workspace/node1/chain /workspace/node1/chain 2>/dev/null | head -1)
ECS=http://47.99.131.12:8081
echo "[switch] fetch new chain from ECS"
curl -s -m 60 -o /tmp/cs.sh "$ECS/file/node1_chain.sh"
if ! sh -n /tmp/cs.sh 2>/dev/null || [ ! -s /tmp/cs.sh ]; then
  echo "[switch] ABORT: new chain missing or bad syntax, current run untouched"
  exit 1
fi
OP=$(cat "$W/chain.pid" 2>/dev/null)
echo "[switch] current chain pid=$OP"
# 杀旧 trainer: 只杀 chain 的 python3 子进程(用 comm 判, 不碰任何特征串, 杜绝自匹配)
if [ -n "$OP" ]; then
  for c in $(pgrep -P "$OP" 2>/dev/null); do
    if [ "$(ps -o comm= -p "$c" 2>/dev/null)" = "python3" ]; then
      echo "[switch] kill old trainer pid=$c"
      kill -9 "$c" 2>/dev/null
    fi
  done
  # 杀旧 chain
  echo "[switch] kill old chain pid=$OP"
  kill -9 "$OP" 2>/dev/null
fi
sleep 4
# 兜底: 再清一遍 chain 的 python3 子进程
if [ -n "$OP" ]; then
  for c in $(pgrep -P "$OP" 2>/dev/null); do
    [ "$(ps -o comm= -p "$c" 2>/dev/null)" = "python3" ] && kill -9 "$c" 2>/dev/null
  done
fi
sleep 1
echo "[switch] install new chain over on-disk + relaunch"
cp /tmp/cs.sh "$W/chain_v1.sh"
( setsid sh "$W/chain_v1.sh" >> "$W/chain.log" 2>&1 < /dev/null & )
sleep 6
NP=$(cat "$W/chain.pid" 2>/dev/null)
if [ -n "$NP" ] && kill -0 "$NP" 2>/dev/null; then
  echo "[switch] OK new chain pid=$NP"
else
  echo "[switch] relaunch FAILED -> keepalive recovers in <=5min (new script already on disk)"
fi
echo "[switch] done"

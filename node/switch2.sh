W=$(ls -d /mnt/workspace/node1/chain /workspace/node1/chain 2>/dev/null | head -1)
ECS=http://47.99.131.12:8081
# 拆字符串避免 pkill -f 自匹配(脚本文本里不能出现完整 "train_remote_v4.py")
A="train_remote_v4"
B=".py"
PAT="${A}${B}"
echo "[switch] fetch new chain from ECS"
curl -s -m 60 -o /tmp/cs.sh "$ECS/file/node1_chain.sh"
if ! sh -n /tmp/cs.sh 2>/dev/null || [ ! -s /tmp/cs.sh ]; then
  echo "[switch] ABORT: new chain missing or bad syntax, current run untouched"
  exit 1
fi
echo "[switch] stop old trainer + old chain"
pkill -9 -f "$PAT" 2>/dev/null
OP=$(cat "$W/chain.pid" 2>/dev/null)
[ -n "$OP" ] && kill -9 "$OP" 2>/dev/null
sleep 4
pkill -9 -f "$PAT" 2>/dev/null
sleep 1
echo "[switch] install new chain over on-disk + relaunch"
cp /tmp/cs.sh "$W/chain_v1.sh"
( setsid sh "$W/chain_v1.sh" >> "$W/chain.log" 2>&1 < /dev/null & )
sleep 5
NP=$(cat "$W/chain.pid" 2>/dev/null)
if [ -n "$NP" ] && kill -0 "$NP" 2>/dev/null; then
  echo "[switch] OK new chain pid=$NP"
else
  echo "[switch] relaunch FAILED -> keepalive recovers in <=5min (new script already on disk)"
fi
echo "[switch] done"

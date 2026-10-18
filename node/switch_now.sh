W=$(ls -d /mnt/workspace/node1/chain /workspace/node1/chain 2>/dev/null | head -1)
ECS=http://47.99.131.12:8081
echo "[switch $(date '+%m-%d %H:%M:%S')] fetch new chain from ECS"
curl -s -m 60 -o /tmp/chain_switch.sh "$ECS/file/node1_chain.sh"
if ! sh -n /tmp/chain_switch.sh 2>/dev/null || [ ! -s /tmp/chain_switch.sh ]; then
  echo "[switch] ABORT: new chain missing or bad syntax, keep current running"
  exit 1
fi
echo "[switch] new chain OK, killing old trainer + old chain"
pkill -9 -f train_remote_v4.py 2>/dev/null
OP=$(cat "$W/chain.pid" 2>/dev/null)
[ -n "$OP" ] && kill -9 "$OP" 2>/dev/null
sleep 4
pkill -9 -f train_remote_v4.py 2>/dev/null
sleep 1
echo "[switch] installing new chain over on-disk + relaunch (threads=8)"
cp /tmp/chain_switch.sh "$W/chain_v1.sh"
( setsid sh "$W/chain_v1.sh" >> "$W/chain.log" 2>&1 < /dev/null & )
sleep 5
NP=$(cat "$W/chain.pid" 2>/dev/null)
if [ -n "$NP" ] && kill -0 "$NP" 2>/dev/null; then
  echo "[switch] OK new chain pid=$NP"
else
  echo "[switch] relaunch FAILED -> keepalive recovers in <=5min"
fi
echo "[switch] done"

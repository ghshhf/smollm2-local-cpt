W=$(ls -d /mnt/workspace/node1/chain /workspace/node1/chain 2>/dev/null | head -1)
echo "W=$W"
echo "--- done ---"; cat "$W/done_steps.txt" 2>/dev/null
echo "--- chain.pid ---"; P=$(cat "$W/chain.pid" 2>/dev/null); echo "pid=$P"; { [ -n "$P" ] && kill -0 "$P" 2>/dev/null && echo "chain ALIVE"; } || echo "chain DEAD"
echo "--- trainer_alive ---"; (ps -eo args 2>/dev/null | grep -v grep | grep -q "train_remote_v4.py" && echo "trainer ALIVE") || echo "trainer DEAD"
echo "--- last 6 train lines ---"; grep "\[train\] step" "$W/chain.log" 2>/dev/null | tail -6
echo "--- last 8 log lines ---"; tail -8 "$W/chain.log" 2>/dev/null
echo "--- heartbeat ---"; cat "$W/heartbeat" 2>/dev/null

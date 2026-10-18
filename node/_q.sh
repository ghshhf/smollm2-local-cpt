W=$(ls -d /mnt/workspace/node1/chain /workspace/node1/chain 2>/dev/null | head -1)
{
echo "=== time ==="; date
echo "W=$W"
echo "--- done_steps ---"; cat "$W/done_steps.txt" 2>/dev/null
echo "--- heartbeat ---"; cat "$W/heartbeat" 2>/dev/null
echo "--- chain.pid ---"; P=$(cat "$W/chain.pid" 2>/dev/null); echo "pid=$P"; { [ -n "$P" ] && kill -0 "$P" 2>/dev/null && echo "chain ALIVE"; } || echo "chain DEAD"
echo "--- trainer ---"; (ps -eo args 2>/dev/null | grep -v grep | grep -q "train_remote" && echo "trainer ALIVE") || echo "trainer DEAD"
echo "--- train steps (last 8) ---"; grep -E "\[train\] step|\[val\]" "$W/chain.log" 2>/dev/null | tail -8
echo "--- log tail ---"; tail -10 "$W/chain.log" 2>/dev/null
echo "--- ls workspace ---"; ls -la "$W" 2>/dev/null | head -20
} > /mnt/workspace/node1/_q_out.txt 2>&1

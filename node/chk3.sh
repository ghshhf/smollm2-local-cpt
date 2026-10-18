W=$(ls -d /mnt/workspace/node1/chain /workspace/node1/chain 2>/dev/null | head -1)
echo "=== chain.pid ==="; cat "$W/chain.pid" 2>/dev/null
echo "=== trainer_alive ==="; (ps -eo args 2>/dev/null | grep -v grep | grep -q "train_remote_v4.py" && echo "trainer ALIVE") || echo "trainer DEAD"
echo "=== set_num_threads ==="; grep "set_num_threads" "$W/chain.log" 2>/dev/null | tail -2
echo "=== last train header ==="; grep "token/步" "$W/chain.log" 2>/dev/null | tail -1
echo "=== last 4 train steps ==="; grep "\[train\] step" "$W/chain.log" 2>/dev/null | tail -4
echo "=== done ==="; cat "$W/done_steps.txt" 2>/dev/null
echo "=== uptime ==="; uptime 2>/dev/null

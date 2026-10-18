W=$(ls -d /mnt/workspace/node1/chain /workspace/node1/chain 2>/dev/null | head -1)
echo "=== LATEST train step ==="; grep "\[train\] step" "$W/chain.log" 2>/dev/null | tail -1
echo "=== last 12 train lines ==="; grep "\[train\] step" "$W/chain.log" 2>/dev/null | tail -12
echo "=== run_tmp ckpt ==="; ls -la "$W/run_tmp/checkpoint-latest" 2>/dev/null | head -20
echo "=== run_tmp step.txt ==="; cat "$W/run_tmp/checkpoint-latest/step.txt" 2>/dev/null
echo "=== R5 cache ==="; ls -la /root/c_cache_R5.npy 2>/dev/null
echo "=== disk /mnt ==="; df -h /mnt/workspace 2>/dev/null | tail -1
echo "=== uptime ==="; uptime 2>/dev/null

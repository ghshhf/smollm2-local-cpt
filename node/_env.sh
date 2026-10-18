W=/mnt/workspace/node1/chain
echo "=== lock/stop ==="; cat $W/chain.pid 2>/dev/null; ls $W/stop 2>/dev/null && echo "!! STOP FLAG 存在" || echo "no stop flag"
echo "=== 关键文件 ==="; ls -la $W/chain_v1.sh $W/chain_v2.sh $W/train_remote_v4.py $W/expand_300m_node1.py 2>&1 | head -8
echo "=== cache ==="; ls -la $W/cache_*.npy 2>/dev/null; ls -la /root/c_cache_*.npy 2>/dev/null || echo "no /root cache"
echo "=== base_tok ==="; ls $W/base_tok/ 2>/dev/null | head -5
echo "=== ckpt ==="; ls -la $W/ckpt/ 2>/dev/null; echo -n "step="; cat $W/ckpt/step.txt 2>/dev/null
echo "=== python/torch ==="; python3 -V; python3 -c "import torch;print(\"torch\",torch.__version__)" 2>&1 | tail -1
echo "=== corpus_reserve ==="; ls -la $W/corpus_reserve/*.jsonl 2>/dev/null | grep -v raw | grep -v pre_wiki
echo "=== chain_v1 log tail ==="; tail -6 $W/chain_v1.log 2>/dev/null

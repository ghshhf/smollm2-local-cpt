#!/bin/sh
# fixrun2.sh — node1: 杀崩坏循环, 拉修复版脚本+convert_st, 转 safetensors, RESET+fp32 训练
W=/mnt/workspace/node1/chain
ECS=http://47.99.131.12:8081
pkill -9 -f chain_v1.sh 2>/dev/null; pkill -9 -f train_remote_v4.py 2>/dev/null; sleep 3
cd "$W"
curl -s -o train_remote_v4.py "$ECS/file/train_remote_v4.py"
curl -s -o chain_v1.sh "$ECS/file/node1_chain.sh"
curl -s -o convert_st.py "$ECS/file/convert_st.py"
python3 -m py_compile train_remote_v4.py && echo "train ok" || { echo "train BAD"; exit 2; }
echo "=== converting base -> safetensors ==="
python3 convert_st.py
echo "=== launching RESET chain (fp32) ==="
(nohup env RESET=1 sh chain_v1.sh > "$W/chain_fixrun2.log" 2>&1 < /dev/null &)
sleep 12
ps aux | grep -E 'train_remote_v4|chain_v1|convert' | grep -v grep
echo FIXRUN2_LAUNCHED

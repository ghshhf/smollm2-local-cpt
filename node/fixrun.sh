#!/bin/sh
# fixrun.sh — node1: 杀掉崩坏循环, 替换修复版 train_remote_v4.py, 启动干净链(无需 RESET, ckpt 已是标准基座 done=0)
W=/mnt/workspace/node1/chain
ECS=http://47.99.131.12:8081
pkill -9 -f chain_v1.sh 2>/dev/null; pkill -9 -f train_remote_v4.py 2>/dev/null; sleep 3
rm -f "$W/train_remote_v4.py"
curl -s -o "$W/train_remote_v4.py" "$ECS/file/train_remote_v4.py"
python3 -m py_compile "$W/train_remote_v4.py" && echo "train compiles OK" || { echo "train BAD"; exit 2; }
(nohup sh "$W/chain_v1.sh" > "$W/chain_fixrun.log" 2>&1 < /dev/null &)
sleep 10
echo "=== procs ==="; ps aux | grep -E 'train_remote_v4|chain_v1' | grep -v grep
echo FIXRUN_LAUNCHED

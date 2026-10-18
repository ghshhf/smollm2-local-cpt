#!/bin/sh
# repair_chain.sh — node1: 补齐 chain/train 脚本(带重试+校验) + RESET 启动 fp32 R4 链
# trap EXIT 保证无论成败都回传 repair.log 到 ECS。
W=/mnt/workspace/node1/chain
ECS=http://47.99.131.12:8081
cd "$W"
LOG=$W/repair.log
: > "$LOG"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

upload_log(){
  { echo REPAIR_DONE; cat "$LOG"; } | curl -s -X POST "$ECS/upload?name=status2_node1.txt&key=k0u4mail" --data-binary @-
  echo UPLOADED
}
trap upload_log EXIT

log "=== repair start ==="

# 0. 杀掉可能残留的旧 chain/train(崩坏循环), 避免双链抢 W/ckpt
pkill -9 -f chain_v1.sh 2>/dev/null; pkill -9 -f train_remote_v4.py 2>/dev/null
sleep 3
log "killed stale procs (if any)"

# 1. 带重试 + 校验 重下 chain_v1.sh / train_remote_v4.py
dl(){ f=$1; u=$2; n=0
  while [ $n -lt 15 ]; do
    curl -s -m120 -o "$f" "$u"
    sz=$(wc -c < "$f" 2>/dev/null || echo 0)
    if [ "$sz" -gt 1000 ]; then log "got $f sz=$sz"; return 0; fi
    n=$((n+1)); sleep 5
  done
  log "FAIL downloading $f"; return 1
}
dl chain_v1.sh "$ECS/file/node1_chain.sh" || exit 1
dl train_remote_v4.py "$ECS/file/train_remote_v4.py" || exit 1
sh -n chain_v1.sh && log "chain_v1 syntax ok" || { log "chain_v1 syntax BAD"; exit 2; }
python3 -m py_compile train_remote_v4.py && log "train_remote_v4 compiles" || { log "train_remote_v4 BAD"; exit 2; }

# 2. 校验标准基座
if [ ! -s r4base_std/pytorch_model.bin ]; then log "ERROR base missing"; exit 3; fi
log "base size=$(wc -c < r4base_std/pytorch_model.bin)"

# 3. 启动链 RESET=1 (清废权重 + 回标准 200M + fp32 训练)
log "launching chain RESET=1"
(nohup env RESET=1 sh chain_v1.sh > chain_reset.log 2>&1 < /dev/null &)
sleep 45
log "=== procs after 45s ==="
ps aux | grep -E 'train_remote_v4|chain_v1' | grep -v grep >> "$LOG"
log "=== corpus dirs ==="
ls -la "$W" | grep corpus >> "$LOG" 2>/dev/null
log "=== chain.log tail ==="
tail -c 1800 chain.log >> "$LOG" 2>/dev/null
log "=== done_steps ==="; cat done_steps.txt >> "$LOG" 2>/dev/null; echo >> "$LOG"
log "=== REPAIR finished, chain should now be training fp32 ==="

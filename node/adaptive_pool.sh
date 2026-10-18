#!/usr/bin/env bash
# adaptive_pool.sh — 自适应并行工作池（内存+负载双门控）
# 用法: bash adaptive_pool.sh <tag> <threads> <alpha1> <alpha2> ...
# 规则: 每放行一个 worker 前检查 ①可用内存>=4GB ②load<=48；不够就等 30s 再查。
#       worker 跑完一个才补下一个 -> 永不 OOM、CPU 自动贴着上限走。
set -u
REPO=/mnt/workspace/node1/gh_repo
ECS="http://47.99.131.12:8081"; UP="k0u4mail"
TAG=$1; TH=$2; shift 2
LOCK=/tmp/adaptive_${TAG}.lock
mkdir "$LOCK" 2>/dev/null || { echo "pool [$TAG] already running"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

cd "$REPO" || exit 1
# 模型优先用 /dev/shm（内存盘，免 NAS 读盘风暴）
if [ -f /dev/shm/smollm/pytorch_model.bin ]; then
  export SMOL_CKPT=/dev/shm/smollm/pytorch_model.bin
  export SMOL_BASE=/dev/shm/smollm/base
else
  export SMOL_CKPT="$REPO/backups/0.1B/0.1B_extracted/model/pytorch_model.bin"
  export SMOL_BASE="$REPO/backups/0.1B/0.1B_extracted/base"
fi
export OMP_NUM_THREADS=$TH
MIN_AVAIL_MB=2500
MAX_LOAD=56
BATCH=${BATCH:-8}

echo "=== adaptive pool [$TAG] start $(date): $* ==="
for a in "$@"; do
  while :; do
    AV=$(free -m | sed -n '2p' | awk '{print $7}')
    LD=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
    if [ "${AV:-0}" -ge "$MIN_AVAIL_MB" ] && [ "${LD:-0}" -le "$MAX_LOAD" ]; then break; fi
    sleep 30
  done
  echo "[pool] launch alpha=$a  (avail=${AV}MB load=$LD) $(date +%H:%M:%S)"
  python3 eval/v1_02_alpha_ladder.py --alpha "$a" --threads "$TH" --batch "$BATCH" \
      --out "eval/v1_02_shard_${TAG}_$a.json" > "eval/v1_02_shard_${TAG}_$a.log" 2>&1 &
  sleep 20   # 等它把模型加载进内存，再决定放不放下一个
done
wait

python3 eval/v1_02_merge.py "$REPO/eval/v1_02_${TAG}.json" "eval/v1_02_shard_${TAG}_*.json" \
    > "$REPO/eval/v1_02_merge_${TAG}.log" 2>&1
echo "--- merge [$TAG] ---"; cat "$REPO/eval/v1_02_merge_${TAG}.log"
curl -s --data-binary @"$REPO/eval/v1_02_${TAG}.json" "$ECS/upload?name=exp_v1_02_${TAG}.json&key=$UP" >/dev/null 2>&1
echo "V1_02_${TAG}_DONE $(date)"

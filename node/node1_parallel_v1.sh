#!/usr/bin/env bash
# V1-02 分片并行驱动：任意 alpha 列表，各起一个进程（吃满多核），跑完自动合并
# 用法: bash parallel_v1.sh <threads> <tag> <alpha1> <alpha2> ...
set -u
REPO=/mnt/workspace/node1/gh_repo
ECS="http://47.99.131.12:8081"; UP="k0u4mail"
TH=${1:-2}; shift
TAG=${1:-coarse}; shift
ALPHAS=${*:-"1.0 0.9 0.7 0.4 0.2 0.0"}
cd "$REPO" || exit 1
export SMOL_CKPT="$REPO/backups/0.1B/0.1B_extracted/model/pytorch_model.bin"
export SMOL_BASE="$REPO/backups/0.1B/0.1B_extracted/base"
export OMP_NUM_THREADS=$TH

rm -f eval/v1_02_shard_${TAG}_*.json
echo "=== parallel V1-02 [$TAG]: $ALPHAS  (threads=$TH) $(date) ==="
n=0
for a in $ALPHAS; do
  python3 eval/v1_02_alpha_ladder.py --alpha $a --threads $TH \
      --out "eval/v1_02_shard_${TAG}_$a.json" > "eval/v1_02_shard_${TAG}_$a.log" 2>&1 &
  n=$((n+1))
done
echo "launched $n shards"
wait

python3 eval/v1_02_merge.py "$REPO/eval/v1_02_${TAG}.json" "eval/v1_02_shard_${TAG}_*.json" \
    > "$REPO/eval/v1_02_merge_${TAG}.log" 2>&1
echo "--- merge [$TAG] ---"
cat "$REPO/eval/v1_02_merge_${TAG}.log"
curl -s --data-binary @"$REPO/eval/v1_02_${TAG}.json" "$ECS/upload?name=exp_v1_02_${TAG}.json&key=$UP" >/dev/null 2>&1
echo "V1_02_${TAG}_DONE $(date)"

#!/usr/bin/env bash
# node1 端实验执行器（由本地经 ECS /file 投递，再经 /task 拉起）
# 流程: 重组 0.1B(本地已有分片) -> 从 ECS 拉 eval 脚本 -> 跑 V1-02 + E1 -> 每步回传状态 + 结果
# 幂等: 两份结果已存在则跳过运行只回传; 串行队列下重派不会并发。
set -u
ECS="http://47.99.131.12:8081"
UP="k0u4mail"
REPO="/mnt/workspace/node1/gh_repo"
PY="$(command -v python3 || command -v python)"
TS="$(date +%s)"
STATUS="$REPO/eval/exp_status_${TS}.txt"

report() { curl -s --data-binary @"$STATUS" "$ECS/upload?name=exp_status.txt&key=$UP" >/dev/null 2>&1 || true; }

echo "=== exp runner start $(date) ===" | tee -a "$STATUS"; report

cd "$REPO" || { echo "REPO_MISSING" | tee -a "$STATUS"; report; exit 1; }

V1_RES="$REPO/eval/v1_02_alpha_ladder.json"
E1_RES="$REPO/eval/e1_result.json"

# 幂等: 结果都齐了就跳过运行
if [ -f "$V1_RES" ] && [ -f "$E1_RES" ]; then
  echo "IDEMPOTENT_SKIP: results exist, re-upload only" | tee -a "$STATUS"
  curl -s --data-binary @"$V1_RES" "$ECS/upload?name=exp_v1_02_result.json&key=$UP" | tee -a "$STATUS"; echo "" | tee -a "$STATUS"
  curl -s --data-binary @"$E1_RES" "$ECS/upload?name=exp_e1_result.json&key=$UP" | tee -a "$STATUS"; echo "" | tee -a "$STATUS"
  echo "SKIP_DONE $(date)" | tee -a "$STATUS"; report; exit 0
fi

# 1) 重组 0.1B（node1 本地已有 8 分片 + reassemble.py，离线即可）
EXT="$REPO/backups/0.1B/0.1B_extracted"
if [ ! -f "$EXT/model/pytorch_model.bin" ]; then
  echo "reassemble 0.1B..." | tee -a "$STATUS"; report
  ( cd "$REPO/backups/0.1B" && python3 reassemble.py ) >> "$STATUS" 2>&1 || { echo "REASSEMBLE_FAIL" | tee -a "$STATUS"; report; exit 3; }
fi
CKPT="$EXT/model/pytorch_model.bin"
BASE="$EXT/base"
[ -f "$CKPT" ] || { echo "CKPT_MISSING $CKPT" | tee -a "$STATUS"; report; exit 4; }
[ -d "$BASE" ] || { echo "BASE_MISSING $BASE" | tee -a "$STATUS"; report; exit 5; }
echo "ckpt=$CKPT base=$BASE" | tee -a "$STATUS"; report

# 2) 从 ECS 拉取本次投递的 eval 脚本（文件名轮换，避免 ECS 锁）
E1_NAME="n1_1792005800_e1_subliminal_proxy.py"
V1_NAME="n1_1792005800_v1_02_alpha_ladder.py"
curl -s "$ECS/file/$E1_NAME?key=$UP" -o "$REPO/eval/e1_subliminal_proxy.py"
curl -s "$ECS/file/$V1_NAME?key=$UP" -o "$REPO/eval/v1_02_alpha_ladder.py"
echo "scripts fetched (e1=$(wc -c < "$REPO/eval/e1_subliminal_proxy.py")B v1=$(wc -c < "$REPO/eval/v1_02_alpha_ladder.py")B)" | tee -a "$STATUS"; report

# 3) 跑实验（限线程，别把 R5 训练拖死）
export SMOL_CKPT="$CKPT"
export SMOL_BASE="$BASE"
export OMP_NUM_THREADS=2
cd "$REPO"
python3 eval/v1_02_alpha_ladder.py --threads 2 --out "$V1_RES" > "$REPO/eval/v1_02_run.log" 2>&1
echo "V1_02 rc=$? $(date)" | tee -a "$STATUS"; report
python3 eval/e1_subliminal_proxy.py > "$REPO/eval/e1_run.log" 2>&1
echo "E1 rc=$? $(date)" | tee -a "$STATUS"; report

# 4) 结果回传（固定名，单次上传不中断则无锁风险）
curl -s --data-binary @"$V1_RES" "$ECS/upload?name=exp_v1_02_result.json&key=$UP" | tee -a "$STATUS"; echo "" | tee -a "$STATUS"
curl -s --data-binary @"$E1_RES" "$ECS/upload?name=exp_e1_result.json&key=$UP" | tee -a "$STATUS"; echo "" | tee -a "$STATUS"
echo "ALL_DONE $(date)" | tee -a "$STATUS"; report

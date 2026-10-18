#!/usr/bin/env bash
# node1 实验执行器 v2：清卡死进程 -> 0.1B 重组 -> V1-02 + E1 并行(线程拉满) -> 每90s回传进度
set -u
ECS="http://47.99.131.12:8081"
UP="k0u4mail"
REPO="/mnt/workspace/node1/gh_repo"
LOG="$REPO/eval"
mkdir -p "$LOG"

# 0) 环境自检：torch 可用？
if ! python3 -c "import torch" 2>/dev/null; then
  echo "NO_TORCH in $(command -v python3)"
  which python python3; ls /mnt/workspace/node1/ 2>/dev/null | head
  { find /mnt/workspace -maxdepth 3 -name "venv" -o -maxdepth 3 -name "*.venv" 2>/dev/null | head; } > /tmp/torch_hunt.txt
  cat /tmp/torch_hunt.txt
  exit 9
fi

# 1) 清掉可能卡死轮询器的旧进程（重组/旧eval），绝不碰 chain trainer
pkill -f "reassemble" 2>/dev/null
pkill -f "v1_02_alpha_ladder" 2>/dev/null
pkill -f "e1_subliminal" 2>/dev/null
sleep 1
echo "=== v2 start $(date) nproc=$(nproc) load=$(cat /proc/loadavg) ==="

# 2) 重组 0.1B（幂等 + 省内存：zip 已校验则直接解压，绝不重拼避免 OOM）
EXT="$REPO/backups/0.1B/0.1B_extracted"
ZIP="$REPO/backups/0.1B/0.1B.zip"
MANIFEST="$REPO/backups/0.1B/MANIFEST.json"
if [ ! -f "$EXT/model/pytorch_model.bin" ]; then
  if [ -f "$ZIP" ]; then
    CHK=$(sha256sum "$ZIP" 2>/dev/null | awk '{print $1}')
    EXP=$(python3 -c "import json;print(json.load(open('$MANIFEST'))['combined_sha256'])" 2>/dev/null)
    if [ "$CHK" = "$EXP" ]; then
      echo "zip 已校验(${CHK:0:12}...) -> 直接解压(不重拼, 省内存)"
      rm -rf "$EXT"; mkdir -p "$EXT"
      unzip -oq "$ZIP" -d "$EXT" && echo "解压完成" || ( echo "unzip失败, 退回reassemble"; ( cd "$REPO/backups/0.1B" && python3 reassemble.py ) ) || { echo "REASSEMBLE_FAIL"; exit 3; }
    else
      echo "zip sha 不符, 重新拼合"; ( cd "$REPO/backups/0.1B" && python3 reassemble.py ) || { echo "REASSEMBLE_FAIL"; exit 3; }
    fi
  else
    echo "无 zip, 拼合"; ( cd "$REPO/backups/0.1B" && python3 reassemble.py ) || { echo "REASSEMBLE_FAIL"; exit 3; }
  fi
fi
CKPT="$EXT/model/pytorch_model.bin"
BASE="$EXT/base"
[ -f "$CKPT" ] && [ -d "$BASE" ] || { echo "CKPT/BASE_MISSING"; ls -la "$EXT" 2>/dev/null; exit 4; }
echo "ckpt=$CKPT base=$BASE OK"

# 3) 拉最新 eval 脚本（v2 投递版：线程放开）
curl -s "$ECS/file/n1_1792006634_e1_subliminal_proxy.py?key=$UP" -o "$LOG/e1_subliminal_proxy.py"
curl -s "$ECS/file/n1_1792006634_v1_02_alpha_ladder.py?key=$UP" -o "$LOG/v1_02_alpha_ladder.py"
echo "scripts: e1=$(wc -c < "$LOG/e1_subliminal_proxy.py")B v1=$(wc -c < "$LOG/v1_02_alpha_ladder.py")B"

# 4) 并行跑：各 4 线程，8 核吃满
export SMOL_CKPT="$CKPT"
export SMOL_BASE="$BASE"
export OMP_NUM_THREADS=4
cd "$REPO"
( python3 eval/v1_02_alpha_ladder.py --threads 4 --out "$LOG/v1_02_alpha_ladder.json" > "$LOG/v1_02_run.log" 2>&1; echo "V1_DONE rc=$? $(date)" ) &
P1=$!
( python3 eval/e1_subliminal_proxy.py > "$LOG/e1_run.log" 2>&1; echo "E1_DONE rc=$? $(date)" ) &
P2=$!

# 5) 每 90s 回传进度（固定名 exp_progress.txt），直到两个都结束
while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do
  {
    echo "=== $(date) load=$(cat /proc/loadavg) ==="
    echo "--- alive: v1=$([ kill -0 $P1 2>/dev/null ] && echo Y || echo N) e1=$([ kill -0 $P2 2>/dev/null ] && echo Y || echo N) ---"
    echo "--- v1 log ---"; tail -n 4 "$LOG/v1_02_run.log" 2>/dev/null
    echo "--- e1 log ---"; tail -n 4 "$LOG/e1_run.log" 2>/dev/null
  } > /tmp/exp_progress.txt 2>&1
  curl -s --data-binary @/tmp/exp_progress.txt "$ECS/upload?name=exp_progress.txt&key=$UP" >/dev/null 2>&1
  sleep 90
done
wait $P1 $P2

# 6) 结果回传
curl -s --data-binary @"$LOG/v1_02_alpha_ladder.json" "$ECS/upload?name=exp_v1_02_result.json&key=$UP" >/dev/null 2>&1
curl -s --data-binary @"$LOG/e1_result.json" "$ECS/upload?name=exp_e1_result.json&key=$UP" >/dev/null 2>&1
{ echo "ALL_DONE $(date)"; echo "--- v1 json ---"; head -c 800 "$LOG/v1_02_alpha_ladder.json" 2>/dev/null; echo; echo "--- e1 json ---"; head -c 800 "$LOG/e1_result.json" 2>/dev/null; } > /tmp/exp_progress.txt
curl -s --data-binary @/tmp/exp_progress.txt "$ECS/upload?name=exp_progress.txt&key=$UP" >/dev/null 2>&1
echo "ALL_DONE"

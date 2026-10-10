#!/bin/bash
# SmolLM2-135M 持续预训练「单 chunk 看门狗启动器」
# 用法: bash run_smollm2_chunk.sh
# - 已完成(done.flag)则跳过
# - 训练中(train.pid 存活)则跳过
# - 否则用 nohup+disown 拉起一个 ~15 分钟的 chunk(--resume 续训, 到点存盘退出)
# 配合每小时自动化 -> 进程被杀自动续, 实现"全自动训练"
cd "D:/work/_local_ai/train_15b" || exit 1
OUT="out_smollm2_135m_pt"
PY="./venv/Scripts/python.exe"

if [ -f "$OUT/done.flag" ]; then
  echo "$(date '+%H:%M:%S') [watchdog] done.flag 存在, 训练已完成, 跳过"
  exit 0
fi

if [ -f "$OUT/train.pid" ]; then
  PID=$(cat "$OUT/train.pid" 2>/dev/null)
  if [ -n "$PID" ] && tasklist //FI "PID eq $PID" 2>/dev/null | grep -qi python.exe; then
    echo "$(date '+%H:%M:%S') [watchdog] 进程 $PID 仍在跑, 跳过"
    exit 0
  fi
fi

echo "$(date '+%H:%M:%S') [watchdog] 拉起新 chunk (--resume --chunk_seconds 900)"
nohup "$PY" train_smollm2_pt.py \
  --resume --chunk_seconds 900 --save_steps 10 --epochs 1 \
  --output_dir "$OUT" --corpus pretrain_corpus/zh_text_v2.jsonl \
  > _smollm2_pt.log 2>&1 < /dev/null &
disown
echo "$(date '+%H:%M:%S') [watchdog] 已拉起, 子进程 PID=$!"

#!/usr/bin/env bash
# 昨书v2 长训启动器（可续训）：自动从 out_lora_v2 最新 checkpoint 续训
set -e
cd "D:/work/_local_ai/train_15b" || exit 1
export MSYS_NO_PATHCONV=1
VE="D:/work/_local_ai/train_15b/venv/Scripts/python.exe"

CKPT=$(ls -d out_lora_v2/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$CKPT" ]; then
  RESUME="--resume_from_checkpoint $CKPT"
  echo "[launch] 续训自 $CKPT -> max_steps 340"
else
  RESUME=""
  echo "[launch] 全新启动 -> max_steps 340"
fi

"$VE" -u train_lora.py \
  --model_path model/Qwen2.5-1.5B-Instruct \
  --data_path data/role_data_v2.jsonl \
  --output_dir out_lora_v2 \
  --max_steps 320 \
  --lora_r 16 --lora_alpha 32 \
  --max_seq_length 384 --grad_accumulation 4 \
  --logging_steps 5 --save_steps 25 \
  $RESUME 2>&1
echo "[exit] train exit code = $?"

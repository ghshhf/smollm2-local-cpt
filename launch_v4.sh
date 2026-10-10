#!/usr/bin/env bash
# launch_v4.sh —— V4 LoRA 训练启动器（nohup+disown 后台存活）
# 数据: data/role_data_v4.jsonl (1232 条 = v3事实624 + 成稿叙事460 + 代码148)
# 策略: 从基座重新训(不叠 v3), 因已把 v3 事实并入数据集, 防遗忘
# 约 1.8 epoch (1232/4*... 560步); 速率 ~55s/步 => ~8.5h
set -u
cd "D:/work/_local_ai/train_15b" || exit 1
VE="D:/work/_local_ai/train_15b/venv/Scripts/python.exe"
rm -f _train_v4.log
nohup "$VE" -u train_lora.py \
  --model_path model/Qwen2.5-1.5B-Instruct \
  --data_path data/role_data_v4.jsonl \
  --output_dir out_lora_v4 \
  --max_steps 560 \
  --lora_r 16 --lora_alpha 32 \
  --max_seq_length 1024 \
  --per_device_batch_size 1 \
  --grad_accumulation 4 \
  --learning_rate 2e-4 \
  > _train_v4.log 2>&1 < /dev/null &
disown
echo "V4 训练已启动 pid=$!"

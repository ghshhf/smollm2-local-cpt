@echo off
chcp 65001 >nul
cd /d D:\work\_local_ai\train_15b
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
.\venv\Scripts\python.exe -u train_lora.py ^
  --model_path backups\0.3B\polaris-03b_20261018 ^
  --data_path data\code_sft_v1.jsonl ^
  --output_dir out_lora_code_03b ^
  --num_train_epochs 6 ^
  --per_device_batch_size 1 ^
  --grad_accumulation 4 ^
  --learning_rate 2e-4 ^
  --lora_r 16 ^
  --lora_alpha 32 ^
  --lora_dropout 0.05 ^
  --max_seq_length 1024 ^
  --logging_steps 5 ^
  --save_steps 150 ^
  --mask_prompt > logs\code_lora_03b.log 2>&1
echo EXIT=%ERRORLEVEL% >> logs\code_lora_03b.log

@echo off
rem Chat LoRA v2: r6_final base + v1 checkpoint-63 adapter, data=chat_sft_v3.jsonl (325, 北极星 identity, no assistant-speak)
cd /d D:\work\_local_ai\train_15b
"D:\work\_local_ai\train_15b\venv\Scripts\python.exe" -u train_lora.py ^
  --model_path "D:\work\_local_ai\train_15b\backups\0.2B\r6_final_20261016" ^
  --data_path "D:\work\_local_ai\train_15b\data\chat_sft_v3.jsonl" ^
  --output_dir "D:\work\_local_ai\train_15b\out_lora_chat_v2" ^
  --init_adapter "D:\work\_local_ai\train_15b\out_lora_chat_v1\checkpoint-63" ^
  --num_train_epochs 2 --per_device_batch_size 1 --grad_accumulation 2 ^
  --learning_rate 1e-4 --save_steps 60 --logging_steps 5 ^
  >> "D:\work\_local_ai\train_15b\logs\chat_lora_v2.log" 2>&1
echo EXITCODE=%ERRORLEVEL% >> "D:\work\_local_ai\train_15b\logs\chat_lora_v2.log"

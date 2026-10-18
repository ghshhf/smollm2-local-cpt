@echo off
rem Chat LoRA v4: masked-loss fix (loss only on assistant tokens) + zero-duplicate 439-sample corpus
rem Ollama serve is running (SYSTEM task OllamaServe) - no conflict, it is inference-only.
cd /d D:\work\_local_ai\train_15b
"D:\work\_local_ai\train_15b\venv\Scripts\python.exe" train_lora.py --model_path "D:\work\_local_ai\train_15b\backups\0.2B\r6_final_20261016" --data_path "D:\work\_local_ai\train_15b\data\chat_sft_v5.jsonl" --output_dir "D:\work\_local_ai\train_15b\out_lora_chat_v4" --mask_prompt --lora_r 16 --lora_alpha 32 --learning_rate 2e-4 --num_train_epochs 4 --logging_steps 5 --per_device_batch_size 2 --grad_accum 1 --max_seq_length 512 > D:\work\_local_ai\train_15b\logs\chat_lora_v4.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> D:\work\_local_ai\train_15b\logs\chat_lora_v4.log
@echo off
cd /d D:\work\_local_ai\train_15b
set OMP_NUM_THREADS=4
set PYTHONIOENCODING=utf-8
set Q=D:\work\_local_ai\models\models\Qwen--Qwen2.5-0.5B-Instruct\snapshots\master
.\venv\Scripts\python.exe -u train_lora.py ^
 --model_path "%Q%" ^
 --data_path data\code_sft_v1.jsonl ^
 --output_dir out_lora_code_qwen ^
 --num_train_epochs 1 ^
 --per_device_batch_size 2 ^
 --grad_accumulation 1 ^
 --learning_rate 2e-4 ^
 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 ^
 --targets q_proj,k_proj,v_proj,o_proj ^
 --max_seq_length 1024 ^
 --mask_prompt ^
 --logging_steps 10 ^
 --save_steps 100 > logs\code_lora_qwen.log 2>&1

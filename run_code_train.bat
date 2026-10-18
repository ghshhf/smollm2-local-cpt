@echo off
cd /d D:\work\_local_ai\train_15b
set OMP_NUM_THREADS=8
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
venv\Scripts\python.exe -u train_smollm2_pt.py --model_dir "D:\work\_local_ai\train_15b\backups\0.2B\r6_final_20261016" --corpus "D:\work\_local_ai\train_15b\data\corpus_code_v1.jsonl" --output_dir "D:\work\_local_ai\train_15b\models\code_02b_v1" --seq_len 2048 --batch_size 2 --grad_accum 8 --lr 1e-4 --epochs 1 --warmup_steps 8 --save_steps 15 --sched cosine > logs\code_train_v1.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> logs\code_train_v1.log

@echo off
cd /d D:\work\_local_ai\train_15b
set OMP_NUM_THREADS=4
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
D:\work\_local_ai\train_15b\venv\Scripts\python.exe tools\vlm_attach.py --steps 1500 --accum 8 --lr 1e-3 > D:\work\_local_ai\train_15b\logs\vlm_align.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> D:\work\_local_ai\train_15b\logs\vlm_align.log

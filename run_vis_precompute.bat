@echo off
cd /d D:\work\_local_ai\train_15b
set OMP_NUM_THREADS=8
venv\Scripts\python.exe -u tools\vlm_attach.py --precompute > logs\vis_precompute.log 2>&1

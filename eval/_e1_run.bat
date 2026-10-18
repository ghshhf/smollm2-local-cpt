@echo off
cd /d D:\work\_local_ai\train_15b
set SMOL_BASE=D:\work\_local_ai\train_15b\backups\0.2B\base_20261010
set E1_REUSE=1
set E1_THREADS=8
D:\work\_local_ai\train_15b\venv\Scripts\python.exe -u eval\e1_subliminal_proxy.py > eval\e1_run.log 2>&1
echo E1_EXIT=%errorlevel% >> eval\e1_run.log

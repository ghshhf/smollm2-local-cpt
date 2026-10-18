@echo off
cd /d D:\work\_local_ai\train_15b
venv\Scripts\python.exe -u eval\e1_subliminal_proxy.py > eval\e1_run.log 2>&1

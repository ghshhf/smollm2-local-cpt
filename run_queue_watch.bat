@echo off
cd /d D:\work\_local_ai\train_15b
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
D:\work\_local_ai\train_15b\venv\Scripts\python.exe -u tools\queue_after_vlm.py

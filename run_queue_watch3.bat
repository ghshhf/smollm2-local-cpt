@echo off
cd /d D:\work\_local_ai\train_15b
set PYTHONIOENCODING=utf-8
venv\Scripts\python.exe -u tools\queue_after_align.py >> logs\queue_after_align_watch.log 2>&1

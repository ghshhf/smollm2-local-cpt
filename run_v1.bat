@echo off
cd /d D:\work\_local_ai\train_15b
D:\work\_local_ai\train_15b\venv\Scripts\python.exe D:\work\_local_ai\train_15b\eval\v1_02_alpha_ladder.py --threads 4 --out D:\work\_local_ai\train_15b\eval\v1_02_alpha_ladder.json >> D:\work\_local_ai\train_15b\eval\v1_02_sys.log 2>&1

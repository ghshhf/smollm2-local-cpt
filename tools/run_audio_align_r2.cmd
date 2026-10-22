@echo off
chcp 65001 >nul
cd /d D:\work\_local_ai\train_15b
D:\work\_local_ai\train_15b\venv\Scripts\python.exe -u D:\work\_local_ai\train_15b\tools\audio_attach.py --stage align --use-cache --val-speakers S0007 --init D:\work\_local_ai\train_15b\models\audio_03b_align\projector_500.pt --steps 2000 --eval-n 30 --save D:\work\_local_ai\train_15b\models\audio_03b_align >> D:\work\_local_ai\train_15b\logs\audio_align_r2.log 2>&1

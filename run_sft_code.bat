@echo off
cd /d D:\work\_local_ai\train_15b
echo ===== PolarisSFT start %DATE% %TIME% ===== >> logs\sft_code_03b.log
venv\Scripts\python.exe -u tools\sft_full.py --data "data/card_code*.jsonl" --epochs 2 --accum 32 --threads 6 --out models\sft_code_03b --save-every 30 --log-every 5 >> logs\sft_code_03b.log 2>&1
echo ===== PolarisSFT end %DATE% %TIME% rc=%ERRORLEVEL% ===== >> logs\sft_code_03b.log

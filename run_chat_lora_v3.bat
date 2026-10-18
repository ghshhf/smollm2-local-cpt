@echo off
rem Chat LoRA v3 - single-line command (no caret continuations; LF/CRLF safe)
rem data v5 = 439 unique convos, zero duplicates; r16 lr 2e-4 4ep
rem waits for any vlm_attach process to exit (max ~90 min), then trains
cd /d D:\work\_local_ai\train_15b
set TRIES=0
:waitvlm
ping -n 61 127.0.0.1 >nul
powershell -NoProfile -NonInteractive -Command "if ((Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*vlm_attach*' } | Measure-Object).Count -gt 0) { exit 1 } else { exit 0 }"
if errorlevel 1 (
  set /a TRIES+=1
  if %TRIES% LSS 90 goto waitvlm
)
"D:\work\_local_ai\train_15b\venv\Scripts\python.exe" -u train_lora.py --model_path "D:\work\_local_ai\train_15b\backups\0.2B\r6_final_20261016" --data_path "D:\work\_local_ai\train_15b\data\chat_sft_v5.jsonl" --output_dir "D:\work\_local_ai\train_15b\out_lora_chat_v3" --num_train_epochs 4 --per_device_batch_size 1 --grad_accumulation 2 --lora_r 16 --lora_alpha 32 --learning_rate 2e-4 --save_steps 100 --logging_steps 5 >> "D:\work\_local_ai\train_15b\logs\chat_lora_v3.log" 2>&1
echo EXITCODE=%ERRORLEVEL% >> "D:\work\_local_ai\train_15b\logs\chat_lora_v3.log"

@echo off
REM V3 续训（从 checkpoint-160 无损接续，保住已训的 2h52m），完成后自动把主链 Train15BChain
REM 重新启用并拉起，让 V4/V5/V6/V7 继续排队。由 SYSTEM 计划任务调用。
setlocal
set PYTHONIOENCODING=utf-8
cd /d D:\work\_local_ai\train_15b
set PY=D:\work\_local_ai\train_15b\venv\Scripts\python.exe

if exist out_lora_v3\_resume_done.flag del out_lora_v3\_resume_done.flag

echo ===== RESUME V3 START %date% %time% =====
%PY% -u train_lora.py --model_path model/Qwen2.5-1.5B-Instruct --data_path data/role_data_v3.jsonl --output_dir out_lora_v3 --max_steps 320 --lora_r 16 --lora_alpha 32 --max_seq_length 1024 --per_device_batch_size 1 --grad_accumulation 4 --learning_rate 2e-4 --resume_from_checkpoint out_lora_v3\checkpoint-160 >> out_lora_v3\_train_resume.log 2>&1
echo ===== RESUME V3 DONE %date% %time% =====
echo done > out_lora_v3\_resume_done.flag

echo ===== RE-CHAIN %date% %time% =====
schtasks /Change /TN Train15BChain /ENABLE
schtasks /Run /TN Train15BChain
endlocal

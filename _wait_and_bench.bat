@echo off
REM 等所有卡下载完成 -> 暂停训练 -> 批量评测所有兼容卡 -> 恢复训练
REM 全程自动，用户无需干预。日志：_batch_bench.log
cd /d D:\work\_local_ai\train_15b
set PY=D:\work\_local_ai\train_15b\venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8

echo ===== 等待卡下载完成 %date% %time% =====
:WAIT
REM fetch 进程还在跑就继续等（用 python 进程命令行判断）
tasklist /FI "IMAGENAME eq python.exe" /FO CSV 2>nul | findstr /I "fetch_hf_cards" >nul
if not "%errorlevel%"=="1" goto WAIT
timeout /t 20 /nobreak >nul
goto WAIT

echo ===== 卡下载完成，暂停训练腾内存 %date% %time% =====
schtasks /End /TN Train15BChain
timeout /t 15 /nobreak >nul
for /f "tokens=2" %%p in ('tasklist /FI "IMAGENAME eq pythonw.exe" /FO CSV /NH') do taskkill /PID %%p /F>nul 2>&1
for /f "tokens=2" %%p in ('wmic process where "name='python.exe'" get processid^,commandline ^| findstr "train_lora"') do taskkill /PID %%p /F >nul 2>&1
timeout /t 10 /nobreak >nul

echo ===== 开始批量评测 %date% %time% =====
%PY% -u eval\bench_cards_batch.py >> _batch_bench.log 2>&1
echo ===== 评测完成，恢复训练 %date% %time% =====
schtasks /Change /TN Train15BChain /ENABLE
schtasks /Run /TN Train15BChain
echo done > _batch_bench_done.flag
@echo off
REM V3 设定卡实测：基座(抽样14题) vs checkpoint-160(全36题)，关键词+原文贴近度双指标
REM 由 SYSTEM 计划任务 EvalV3Bench 调用，输出落日志文件，避免被会话回收掐断。
setlocal
set PYTHONIOENCODING=utf-8
cd /d D:\work\_local_ai\train_15b
set PY=D:\work\_local_ai\train_15b\venv\Scripts\python.exe

if exist eval\_bench_done.flag del eval\_bench_done.flag

echo ===== BASE START %date% %time% =====
%PY% -u eval/eval_v7.py --data eval/holdout_v3.jsonl --tag base_v3 --limit 14 --max_new_tokens 160 --dtype bf16 > eval\_bench_base.log 2>&1
echo ===== BASE DONE %date% %time% =====

echo ===== V3CK160 START %date% %time% =====
%PY% -u eval/eval_v7.py --data eval/holdout_v3.jsonl --adapter out_lora_v3/checkpoint-160 --tag v3_ck160 --max_new_tokens 160 --dtype bf16 > eval\_bench_v3.log 2>&1
echo ===== V3CK160 DONE %date% %time% =====

echo done > eval\_bench_done.flag
endlocal

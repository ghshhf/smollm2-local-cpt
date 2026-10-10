@echo off
REM 等 EmbeddingGemma 2 下载完成 -> 自动建一张向量索引做验证(取最小的一张卡做样例)。
REM 用 SYSTEM 任务调用，避免被会话回收。日志见 _vec_index.log
cd /d D:\work\_local_ai\train_15b
set PY=D:\work\_local_ai\train_15b\venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8

echo ===== 等待模型下载完成 %date% %time% =====
:WAIT
if exist embedding_model_path.txt goto DONE
timeout /t 20 /nobreak >nul
goto WAIT

:DONE
echo ===== 模型就绪，开始建向量索引 %date% %time% =====
%PY% -u build_vector_index.py --data data/card_reason.jsonl --out vec_reason --truncate 256 --limit 200>> _vec_index.log 2>&1
echo ===== DONE %date% %time% =====
echo done > _vec_index_done.flag
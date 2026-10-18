@echo off
rem VLM v6 - align 续训（2026-10-17 验收失败后决策）
rem  v5(1500步≈0.5epoch)盲测失败：视觉条件化弱。根因=训练量不足。
rem  v6: 从 v5 projector + v5 LLM 权重续训，3000步(累计≈1.5 epoch)，LR 腰斩。
rem  依据：SmolVLM2 配方(LLM联训 v4=2.05 vs v5=1.71 已实证)；
rem        L2 信息天花板——固定 24837 对语料加 epoch 收益递减，1.5 epoch 为诊断上限。
rem  验收：复用 logs/vlm_v5_accept.log 同款盲测(抓拍图+训练图诊断)，禁信 loss。
cd /d D:\work\_local_ai\train_15b
set OMP_NUM_THREADS=8
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
D:\work\_local_ai\train_15b\venv\Scripts\python.exe -u D:\work\_local_ai\train_15b\tools\vlm_attach.py ^
  --steps 3000 --accum 8 --lr 5e-4 --warmup 100 ^
  --ps 2 --train-llm --use-cache ^
  --init "D:\work\_local_ai\train_15b\models\vlm_02b_v5\projector.pt" ^
  --llm "D:\work\_local_ai\train_15b\models\vlm_02b_v5" ^
  --save "D:\work\_local_ai\train_15b\models\vlm_02b_v6" ^
  >> "D:\work\_local_ai\train_15b\logs\vlm_v6.log" 2>&1
echo EXITCODE=%ERRORLEVEL% >> "D:\work\_local_ai\train_15b\logs\vlm_v6.log"

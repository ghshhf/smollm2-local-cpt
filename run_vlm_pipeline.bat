@echo off
cd /d D:\work\_local_ai\train_15b
:wait
if not exist "data\vlm\vis_feats.pt" (
  timeout /t 120 /nobreak > nul
  goto wait
)
venv\Scripts\python.exe -u tools\vlm_attach.py --stage align --steps 3000 --use-cache --vis-tokens 64 --lr 3e-4 --accum 8 --save models\vlm_02b_align > logs\vlm_align_v4_redo.log 2>&1

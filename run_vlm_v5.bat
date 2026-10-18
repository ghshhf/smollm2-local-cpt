@echo off
rem VLM v5 - SmolVLM2 recipe stage A (arXiv 2504.05299):
rem  1) --train-llm: 0.2B LLM full-param training (lr x0.1), vision frozen
rem  2) --ps 2: pixel shuffle 196->49 tokens (small models LIKE token compression)
rem  3) --text-mix 0.14: pure-text samples from chat_sft_v3.jsonl protect language ability
rem  4) use-cache: frozen SigLIP features precomputed (data/vlm/vis_feats.pt)
rem Inference gets repetition-penalty 1.3 + no-repeat-trigram automatically.
cd /d D:\work\_local_ai\train_15b
set OMP_NUM_THREADS=8
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
D:\work\_local_ai\train_15b\venv\Scripts\python.exe -u tools\vlm_attach.py ^
  --steps 1500 --accum 8 --lr 1e-3 --warmup 200 ^
  --ps 2 --train-llm --use-cache ^
  --save "D:\work\_local_ai\train_15b\models\vlm_02b_v5" ^
  >> "D:\work\_local_ai\train_15b\logs\vlm_v5.log" 2>&1
echo EXITCODE=%ERRORLEVEL% >> "D:\work\_local_ai\train_15b\logs\vlm_v5.log"

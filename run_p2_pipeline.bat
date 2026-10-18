@echo off
cd /d D:\work\_local_ai\train_15b
set OMP_NUM_THREADS=4
set PYTHONIOENCODING=utf-8
set Q=D:\work\_local_ai\models\models\Qwen--Qwen2.5-0.5B-Instruct\snapshots\master
set Q15=D:\work\_local_ai\train_15b\model\Qwen2.5-1.5B-Instruct
set PY=.\venv\Scripts\python.exe

echo [P2 %date% %time%] === step1: train Qwen2.5-0.5B coding LoRA (the ONLY training) === >> logs\p2_pipeline.log
%PY% -u train_lora.py ^
 --model_path "%Q%" ^
 --data_path data\code_sft_v1.jsonl ^
 --output_dir out_lora_code_qwen ^
 --num_train_epochs 1 ^
 --per_device_batch_size 2 ^
 --grad_accumulation 1 ^
 --learning_rate 2e-4 ^
 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 ^
 --targets q_proj,k_proj,v_proj,o_proj ^
 --max_seq_length 1024 ^
 --mask_prompt ^
 --logging_steps 10 ^
 --save_steps 100 > logs\code_lora_qwen.log 2>&1
if errorlevel 1 ( echo [P2] training FAILED rc=%errorlevel% >> logs\p2_pipeline.log & exit /b 1 )
echo [P2 %date% %time%] training done >> logs\p2_pipeline.log

echo [P2 %date% %time%] === step2: zero-pad remap trained LoRA -> Qwen2.5-1.5B === >> logs\p2_pipeline.log
%PY% -u tools\module_loader.py port --model "%Q15%" --adapter out_lora_code_qwen --out out_lora_code_qwen_15b >> logs\p2_pipeline.log 2>&1
%PY% -u tools\module_loader.py verify --model "%Q15%" --adapter out_lora_code_qwen --dst out_lora_code_qwen_15b >> logs\p2_pipeline.log 2>&1
echo [P2 %date% %time%] remap done >> logs\p2_pipeline.log

echo [P2 %date% %time%] === step3: code_eval 4-way (bare/mounted x 0.5B/1.5B) === >> logs\p2_pipeline.log
%PY% -u tools\code_eval.py --model "%Q%"  --tag bare05b --dtype fp32 --out logs\code_eval_bare05b.json >> logs\p2_pipeline.log 2>&1
%PY% -u tools\code_eval.py --model "%Q%"  --adapter out_lora_code_qwen --tag codeqwen05b --dtype fp32 --out logs\code_eval_codeqwen05b.json >> logs\p2_pipeline.log 2>&1
%PY% -u tools\code_eval.py --model "%Q15%" --tag bare15b --dtype fp32 --out logs\code_eval_bare15b.json >> logs\p2_pipeline.log 2>&1
%PY% -u tools\code_eval.py --model "%Q15%" --adapter out_lora_code_qwen_15b --tag codeqwen15b_remap --dtype fp32 --out logs\code_eval_codeqwen15b_remap.json >> logs\p2_pipeline.log 2>&1
echo [P2 %date% %time%] === ALL DONE === >> logs\p2_pipeline.log

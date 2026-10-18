#!/bin/sh
# node1 上跑 α 阶梯实验（零训练，纯前向）
cd /mnt/workspace/node1/chain || exit 1
export SMOL_BASE=/mnt/workspace/node1/chain
export SMOL_CKPT=/mnt/workspace/node1/chain/pytorch_model.bin
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
{
echo "#### ENV"; date; nproc; free -g | head -2
python3 -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)"
echo "#### FILES"; ls -la pytorch_model.bin config.json tokenizer.json v1_02_alpha_ladder.py book_yingchuan.jsonl 2>&1 | head -6
echo "#### RUN"
python3 v1_02_alpha_ladder.py --threads 8 --out /mnt/workspace/node1/chain/v1_02_alpha_ladder.json 2>&1 | grep -avE "size mismatch|^\s*$" | tail -30
echo "#### RESULT"
cat v1_02_alpha_ladder.json 2>/dev/null | head -40
} 2>&1 | curl -s -m 1800 -X POST 'http://47.99.131.12:8081/upload?name=ladder_result.txt&key=k0u4mail' --data-binary @-

{ echo "#### BASE"; ls -d /root/r4work/base_next /mnt/workspace/node1/models/SmolLM2-200M 2>/dev/null
  echo "#### TOKENIZER"; find /root/r4work/base_next -name "tokenizer*" 2>/dev/null | head -3
  echo "#### NPROC"; nproc
  echo "#### PY"; which python3; python3 -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)"
  echo "#### SPACE"; df -h /mnt/workspace | tail -1
} 2>&1 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=vp_env.txt&key=k0u4mail' --data-binary @-

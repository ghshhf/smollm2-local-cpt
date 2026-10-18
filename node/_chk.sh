{ echo "#### NPROC"; nproc
  echo "#### MEM"; free -g | head -2
  echo "#### PY"; python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | head -2
  echo "#### TRANSFORMERS"; python3 -c "import transformers;print(transformers.__version__)" 2>&1 | head -1
  echo "#### DISK"; df -h /root /mnt/workspace 2>/dev/null | tail -3
  echo "#### FIND_135M"; find /root /mnt/workspace -maxdepth 4 -name "*135m*" -o -maxdepth 4 -name "*smollm2*" 2>/dev/null | head -6
  echo "#### BASE_NEXT"; ls /root/r4work/base_next 2>/dev/null | head -8
} 2>&1 | curl -s -m 180 -X POST 'http://47.99.131.12:8081/upload?name=chk2.txt&key=k0u4mail' --data-binary @-

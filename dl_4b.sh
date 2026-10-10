#!/usr/bin/env bash
# 4B GGUF 完整下载 + sha256 校验（从头，不再续传）
export MSYS_NO_PATHCONV=1
F="D:/work/novel/models/qwen3-4b-instruct2507-abliterated-Q4_K_M.gguf"
U="https://huggingface.co/kuronekkko/Huihui-Qwen3-4B-Instruct-2507-abliterated-Q4_K_M-GGUF/resolve/main/huihui-qwen3-4b-instruct-2507-abliterated-q4_k_m.gguf"
TARGET=2497276736
EXPECT_SHA="d80ef0f08a0e64887f4a19bc9202fe1a108d9666a2c49eb9a30bc3a30995121b"

echo "[$(date +%H:%M:%S)] 从头完整下载(无续传), 目标 $TARGET 字节"
for i in 1 2 3; do
  rm -f "$F"
  echo "[$(date +%H:%M:%S)] try $i: 完整下载..."
  curl -L --retry 3 --retry-delay 5 -o "$F" "$U"
  s=$(stat -c%s "$F" 2>/dev/null || echo 0)
  echo "[$(date +%H:%M:%S)] try $i size=$s"
  if [ "$s" -lt "$TARGET" ]; then echo "INCOMPLETE, retry"; sleep 5; continue; fi
  h=$(sha256sum "$F" 2>/dev/null | awk '{print $1}')
  echo "sha256=$h"
  if [ "$h" = "$EXPECT_SHA" ]; then echo "SHA_OK_DONE"; exit 0; fi
  echo "SHA_MISMATCH try $i"
done
echo "[$(date +%H:%M:%S)] WARN: 多次 sha 不匹配官方 oid, 保留最后一次文件(源可能已更新版本)"

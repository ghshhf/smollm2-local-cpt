#!/bin/bash
# upload_r6.sh — 拆分 R6 语料并上传到 ECS /file/ (供 node1 中继到 NAS)
cd /d/work/_local_ai/train_15b/pretrain_corpus || exit 1
ECS="http://47.99.131.12:8081"
rm -f r6_*
split -d -b 90000000 zh_text_r6.jsonl r6_
echo "PARTS:"; ls -la r6_*
for f in r6_*; do
  local_sz=$(wc -c < "$f")
  ok=0
  for a in 1 2 3 4 5; do
    curl -s -m600 -X POST "$ECS/upload?name=$f&key=k0u4mail" --data-binary @"$f" >/dev/null 2>&1
    sz=$(curl -s -m60 -o /dev/null -w "%{size_download}" "$ECS/file/$f")
    if [ "$sz" = "$local_sz" ]; then echo "OK $f ($sz) a=$a"; ok=1; break; fi
    echo "RETRY $f a=$a got=$sz want=$local_sz"; sleep 3
  done
  [ "$ok" = 0 ] && echo "FAIL $f"
done
echo "R6_UPLOAD_DONE $(date)"
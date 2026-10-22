#!/bin/bash
# pull_siglip.sh — node1 侧：从 ECS 信箱拉 SigLIP2-base 分片 -> 合并 -> 解包
set -u
ECS="http://47.99.131.12:8081"
KEY="k0u4mail"
WORK=/mnt/workspace/node1/_pack_siglip
DST=/mnt/workspace/node1/models
mkdir -p "$WORK" "$DST"
cd "$WORK" || exit 1

echo "[pull] 开始 $(date +%H:%M:%S)"
t0=$(date +%s)
ok=0
for i in $(seq -w 0 36); do
    name="siglip_p_${i}"
    curl -s --max-time 300 "${ECS}/file/${name}?key=${KEY}" -o "p_${i}"
    sz=$(stat -c%s "p_${i}" 2>/dev/null || echo 0)
    if [ "$sz" -gt 1000000 ]; then
        ok=$((ok + 1))
    else
        echo "  [WARN] ${name} 大小异常=${sz}"
    fi
done
echo "[pull] 拉取完成 ok=${ok}/37 用时 $(( $(date +%s) - t0 ))s"

cat p_* > siglip2base.tar
TS=$(stat -c%s siglip2base.tar)
echo "[pull] tar=${TS} (期望 1535221760)"
if [ "$TS" -ne 1535221760 ]; then
    echo "[fatal] tar 大小不符，中止解包"
    exit 2
fi

tar -xf siglip2base.tar -C "$DST"
echo "[pull] 解包完成:"
ls -la "$DST/siglip2-base/" | head -10
echo "[pull] DONE $(date +%H:%M:%S)"

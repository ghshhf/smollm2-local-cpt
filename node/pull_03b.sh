#!/bin/sh
# pull_03b.sh —— 在 node1 上把 0.3B 主干从 ECS 信箱拼回来
#
# 顺序坑：cat 0*.part 依赖 shell 字典序，00..14 两位补零才安全。
# 所以这里逐个追加，不靠通配符排序。
set -u
ECS="http://47.99.131.12:8081"
KEY="k0u4mail"
WANT_SHA="${1:-}"
DST=/mnt/workspace/node1/models/polaris-03b

mkdir -p "$DST" || exit 1
cd "$DST" || exit 1
echo "[pull] start $(date -u +%FT%TZ) -> $DST"

: > model.safetensors
i=0
while [ "$i" -le 14 ]; do
  n=$(printf "%02d" "$i")
  for att in 1 2 3 4; do
    curl -sS -m 600 -o "$n.part" "$ECS/file/03b_bin.$n?key=$KEY" && [ -s "$n.part" ] && break
    echo "[pull] retry $n att=$att"; sleep 3
  done
  sz=$(wc -c < "$n.part" 2>/dev/null || echo 0)
  echo "[pull] $n  $sz bytes"
  cat "$n.part" >> model.safetensors
  rm -f "$n.part"
  i=$((i + 1))
done

for f in config.json generation_config.json; do
  curl -sS -m 120 -o "$f" "$ECS/file/$f?key=$KEY" || echo "[pull] fail $f"
done

TOT=$(wc -c < model.safetensors)
echo "[pull] model.safetensors = $TOT bytes"
if [ -n "$WANT_SHA" ]; then
  GOT=$(sha256sum model.safetensors | cut -d' ' -f1)
  echo "[pull] sha256 got=$GOT"
  echo "[pull] sha256 want=$WANT_SHA"
  if [ "$GOT" = "$WANT_SHA" ]; then echo "[pull] SHA_OK"; else echo "[pull] SHA_MISMATCH"; fi
fi
echo "[pull] done $(date -u +%FT%TZ)"

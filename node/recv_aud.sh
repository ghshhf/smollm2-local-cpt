#!/bin/sh
# recv_aud.sh — 在 node1 上拉取音频分片、解包、重写路径（幂等，可重复跑）
# 由本机经 ECS 信箱投放到 /mnt/workspace/node1/recv_aud.sh 后执行。
ECS=http://47.99.131.12:8081
KEY=k0u4mail
P=/mnt/workspace/node1/audio_parts
DST=/mnt/workspace/node1/data/audio
L="aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq ar as at au av aw ax ay az ba bb bc bd be bf bg bh bi bj bk bl bm bn bo bp bq br bs"

mkdir -p "$P" "$DST" /mnt/workspace/node1/tools
cd "$P" || exit 1

echo "[recv] 开始拉取 $(echo $L | wc -w) 片"
for s in $L; do
  if [ -s "ap_$s" ]; then
    continue
  fi
  ok=0
  for try in 1 2 3; do
    curl -s -m 600 -o "ap_$s" "$ECS/file/aud_p_$s?key=$KEY"
    if [ -s "ap_$s" ]; then ok=1; break; fi
    sleep 2
  done
  if [ "$ok" = "1" ]; then
    echo "[recv] ap_$s $(wc -c < ap_$s) bytes"
  else
    echo "[recv] FAIL ap_$s"
  fi
done

echo "[recv] 分片数 $(ls ap_* 2>/dev/null | wc -l)  合计 $(du -sh . | cut -f1)"
echo "[recv] 解包..."
cat ap_* | tar xzf - -C "$DST" 2>/dev/null
echo "[recv] 解包后:"
ls "$DST" | head -5
echo "[recv] wav 数: $(find "$DST/wav" -name '*.wav' 2>/dev/null | wc -l)"

# 重写 pairs 的 audio 路径（Windows -> node1）
python3 - "$DST/aishell_pairs.jsonl" <<'PYEOF'
import json, sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print("[paths] pairs 不存在:", p)
    raise SystemExit(1)
OLD = "D:\\work\\_local_ai\\train_15b\\data\\audio"
NEW = "/mnt/workspace/node1/data/audio"
n, miss = 0, 0
out = []
for line in open(p, encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    a = r["audio"].replace(OLD, NEW).replace("\\", "/")
    r["audio"] = a
    if not os.path.exists(a):
        miss += 1
    out.append(r)
    n += 1
if n and miss == 0:
    with open(p, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[paths] 重写 %d 条，缺失 0 -> 已覆盖" % n)
else:
    print("[paths] 重写 %d 条，缺失 %d（未覆盖，需检查）" % (n, miss))
PYEOF

echo "[recv] 下载脚本"
curl -s -m 120 -o /mnt/workspace/node1/tools/audio_attach.py "$ECS/file/aa_script.py?key=$KEY"
ls -l /mnt/workspace/node1/tools/audio_attach.py

echo "[recv] DONE"

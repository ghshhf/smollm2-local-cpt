#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""30s speed test: node1 -> HF xet CDN for L3 ms shard; also read poetry clone log."""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
CMD = (
    "URL='https://hf-mirror.com/datasets/openbmb/Ultra-FineWeb-L3/resolve/main/data/ultrafineweb_zh_l3/multi_style/part-00000-c13afd3b-b5fb-4acd-97dc-e045a844c126-c000.snappy.parquet'; "
    "S=$(date +%s); curl -sL -m 30 -o /tmp/xetspd.bin \"$URL\"; E=$(date +%s); "
    "{ echo \"elapsed=$((E-S))s size=$(stat -c%s /tmp/xetspd.bin 2>/dev/null || echo 0)\"; "
    "rm -f /tmp/xetspd.bin; "
    "echo '--- poetry clone log ---'; tail -6 /mnt/workspace/node1/poetry_clone.log 2>/dev/null; } 2>&1 | "
    "curl -s -X POST 'http://47.99.131.12:8081/upload?name=xet_spd_001.txt&key=k0u4mail' --data-binary @-"
)
body = json.dumps({"cmd": CMD, "id": "xet-spd-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST ->", r.status)
except Exception as e:
    print("ERR:", e)

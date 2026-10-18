#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test node1 -> HF xet CDN reachability with a 10MB range request."""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
CMD = (
    "time curl -sL -r 0-10485760 -m 90 -o /tmp/xettest.bin "
    "'https://hf-mirror.com/datasets/openbmb/Ultra-FineWeb-L3/resolve/main/data/ultrafineweb_zh_l3/qa/part-00000-100cee91-ea2e-4268-9b98-1403cd5d9d11-c000.snappy.parquet' "
    "2>&1; ls -la /tmp/xettest.bin"
)

body = json.dumps({"cmd": CMD, "id": "xet-test-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR:", e)

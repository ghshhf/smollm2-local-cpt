#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dispatch FIXED l3_creative_fill.py (v2) to node1 via ECS /task. Reuses existing 4.6GB jsonl (skips convert)."""
import json, time, base64, urllib.request

SCRIPT = "D:/work/_local_ai/train_15b/node/l3_creative_fill.py"
b64 = base64.b64encode(open(SCRIPT, "rb").read()).decode()
CMD = (
    "echo '%s' | base64 -d > /mnt/workspace/node1/l3_creative_fill.py && "
    "(nohup python3 /mnt/workspace/node1/l3_creative_fill.py "
    "> /mnt/workspace/node1/creative_fill.log 2>&1 </dev/null &) && "
    "sleep 3 && head -5 /mnt/workspace/node1/creative_fill.log"
) % b64

ECS = "http://47.99.131.12:8081"
body = json.dumps({"cmd": CMD, "id": "l3-creative-fill-002", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=30)
    print("POST /task ->", r.status, r.read().decode()[:300])
except Exception as e:
    print("POST ERR:", e)

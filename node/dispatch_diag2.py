#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dispatch diag_creative.py to node1 with ROTATED id+filename (avoid ECS /file cache)."""
import json, time, base64, urllib.request

SCRIPT = "D:/work/_local_ai/train_15b/node/diag_creative.py"
b64 = base64.b64encode(open(SCRIPT, "rb").read()).decode()
# 轮换上传文件名 + 任务 id，避开 ECS 同名缓存
CMD = ("echo '%s' | base64 -d > /mnt/workspace/node1/diag_creative.py && "
       "sed -i 's/diag_creative.txt/diag_creative_002.txt/' /mnt/workspace/node1/diag_creative.py && "
       "python3 /mnt/workspace/node1/diag_creative.py "
       "> /mnt/workspace/node1/diag_creative_002.log 2>&1") % b64

ECS = "http://47.99.131.12:8081"
body = json.dumps({"cmd": CMD, "id": "diag-creative-002", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=30)
    print("POST /task ->", r.status, r.read().decode()[:300])
except Exception as e:
    print("POST ERR:", e)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dispatch diag_creative_now.py to node1 via ECS /task (foreground, quick diag)."""
import json, time, base64, urllib.request

SCRIPT = "D:/work/_local_ai/train_15b/node/diag_creative_now.py"
b64 = base64.b64encode(open(SCRIPT, "rb").read()).decode()
CMD = (
    "echo '%s' | base64 -d > /mnt/workspace/node1/diag_creative_now.py && "
    "python3 /mnt/workspace/node1/diag_creative_now.py"
) % b64

ECS = "http://47.99.131.12:8081"
body = json.dumps({"cmd": CMD, "id": "diag-creative-now", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=30)
    print("POST /task ->", r.status, r.read().decode()[:300])
except Exception as e:
    print("POST ERR:", e)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dispatch UFW-zh download+convert (self-contained b64 sh) to node1."""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
B64 = open("D:/work/_local_ai/train_15b/node/dl_l3.b64").read().strip()
CMD = (
    "echo '%s' | base64 -d > /mnt/workspace/node1/dl_l3.sh && "
    "(nohup bash /mnt/workspace/node1/dl_l3.sh > /mnt/workspace/node1/corpus_04b_l3.log 2>&1 </dev/null &) && "
    "sleep 2 && head -3 /mnt/workspace/node1/corpus_04b_l3.log"
) % B64

body = json.dumps({"cmd": CMD, "id": "l3-dl-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=20)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR:", e)

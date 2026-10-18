#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
B64 = open("D:/work/_local_ai/train_15b/node/conv_poetry.b64").read().strip()
CMD = ("echo '%s' | base64 -d > /mnt/workspace/node1/conv_poetry.py && "
       "python3 /mnt/workspace/node1/conv_poetry.py 2>&1") % B64
body = json.dumps({"cmd": CMD, "id": "poetry-conv-002", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST ->", r.status)
except Exception as e:
    print("ERR:", e)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
B64 = open("D:/work/_local_ai/train_15b/node/conv_poetry_csv.b64").read().strip()
CMD = ("echo '%s' | base64 -d > /mnt/workspace/node1/conv_poetry_csv.py; "
       "{ python3 /mnt/workspace/node1/conv_poetry_csv.py; } 2>&1 | "
       "curl -s -X POST 'http://47.99.131.12:8081/upload?name=poetry_conv_007.txt&key=k0u4mail' --data-binary @-") % B64
body = json.dumps({"cmd": CMD, "id": "poetry-conv-007", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST ->", r.status)
except Exception as e:
    print("ERR:", e)

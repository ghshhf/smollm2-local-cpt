#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dispatch diag with a TIMESTAMPED upload filename to bypass ECS /file same-name cache."""
import json, time, base64, urllib.request

SCRIPT = "D:/work/_local_ai/train_15b/node/diag_creative.py"
b64 = base64.b64encode(open(SCRIPT, "rb").read()).decode()
TS = int(time.time())
# 把上传文件名替换成带时间戳的，确保每次 /file 取到的是新内容
CMD = ("echo '%s' | base64 -d > /mnt/workspace/node1/diag_creative.py && "
       "sed -i 's/diag_creative.txt/diag_%d.txt/' /mnt/workspace/node1/diag_creative.py && "
       "python3 /mnt/workspace/node1/diag_creative.py "
       "> /mnt/workspace/node1/diag_creative.log 2>&1") % (b64, TS)

ECS = "http://47.99.131.12:8081"
body = json.dumps({"cmd": CMD, "id": "diag-creative-%d" % TS, "ts": TS}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=30)
    print("POST /task ->", r.status, r.read().decode()[:200])
    print("EXPECTED_FILE=diag_%d.txt" % TS)
except Exception as e:
    print("POST ERR:", e)

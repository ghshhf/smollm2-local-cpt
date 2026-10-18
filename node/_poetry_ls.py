#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = (
    "{ ls -la /mnt/workspace/node1/corpus_04b/poetry/; echo '--- head train.csv ---'; "
    "head -c 400 /mnt/workspace/node1/corpus_04b/poetry/train.csv; echo; echo '--- wc ---'; "
    "wc -l /mnt/workspace/node1/corpus_04b/poetry/train.csv; } 2>&1 | "
    "curl -s -X POST 'http://47.99.131.12:8081/upload?name=poetry_ls_006.txt&key=k0u4mail' --data-binary @-"
)
body = json.dumps({"cmd": CMD, "id": "poetry-ls-006", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST ->", r.status)
except Exception as e:
    print("ERR:", e)

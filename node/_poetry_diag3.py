#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = (
    "{ head -c 600 /mnt/workspace/node1/corpus_04b/poetry/chinese-poetry-collection.json; echo; echo '---tail---'; "
    "tail -c 300 /mnt/workspace/node1/corpus_04b/poetry/chinese-poetry-collection.json; } 2>&1 | "
    "curl -s -X POST 'http://47.99.131.12:8081/upload?name=poetry_diag_005.txt&key=k0u4mail' --data-binary @-"
)
body = json.dumps({"cmd": CMD, "id": "poetry-diag-005", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST ->", r.status)
except Exception as e:
    print("ERR:", e)

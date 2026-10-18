#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ ls -la /mnt/workspace/node1/train_live_push.py 2>&1; "
       "echo '--- try run 8s ---'; "
       "timeout 8 python3 -u /mnt/workspace/node1/train_live_push.py 2>&1 | head -8; "
       "echo '--- pgrep ---'; pgrep -af train_live 2>&1 | head -3; "
       "echo '--- python3 which ---'; which python3 python 2>&1; } 2>&1 | "
       "curl -s -m90 -X POST 'http://47.99.131.12:8081/upload?name=live_diag_002.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "live-diag-002", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

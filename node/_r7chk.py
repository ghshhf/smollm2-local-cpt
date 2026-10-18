#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ echo '=== now ==='; date '+%H:%M:%S'; "
       "echo '=== chain_v2.log tail 6 ==='; tail -6 /mnt/workspace/node1/chain/chain_v2.log 2>/dev/null; "
       "echo '=== train proc ==='; pgrep -af train_remote_v4 | grep -v grep | head -2; "
       "echo '=== corpus R7 ==='; ls -la /mnt/workspace/node1/chain/corpus_R7/ 2>/dev/null | head -3; } 2>&1 | "
       "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=r7_chk_001.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "r7-chk-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

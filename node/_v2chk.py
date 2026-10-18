#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ echo '=== now ==='; date '+%H:%M:%S'; echo '=== all procs ==='; ps aux | grep -E 'chain|train|expand|python3 -u' | grep -v grep | head -8; "
       "echo '=== chain.log tail 30 ==='; tail -30 /mnt/workspace/node1/chain/chain.log; "
       "echo '=== new logs ==='; ls -lat /mnt/workspace/node1/chain/ | head -8; "
       "echo '=== chain_v2.log? ==='; tail -10 /mnt/workspace/node1/chain/chain_v2.log 2>/dev/null || echo no_v2log; } 2>&1 | "
       "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=v2_chk_001.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "v2-chk-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

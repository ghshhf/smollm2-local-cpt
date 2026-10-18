#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ echo '=== now ==='; date '+%H:%M:%S'; echo '=== chain.log tail 8 ==='; tail -8 /mnt/workspace/node1/chain/chain.log; "
       "echo '=== done_steps ==='; cat /mnt/workspace/node1/chain/done_steps.txt; echo; "
       "echo '=== train proc ==='; pgrep -af train_remote_v4 | grep -v grep | head -2 || echo NO_TRAIN; "
       "echo '=== chain_v1 alive ==='; pgrep -af chain_v1 | grep -v grep | head -2; } 2>&1 | "
       "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=r6_final_003.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "r6-final-003", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

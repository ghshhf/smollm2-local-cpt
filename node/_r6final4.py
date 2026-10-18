#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ echo '=== now ==='; date '+%H:%M:%S'; echo '=== chain.log tail 25 ==='; tail -25 /mnt/workspace/node1/chain/chain.log; "
       "echo '=== done_steps ==='; cat /mnt/workspace/node1/chain/done_steps.txt; echo; "
       "echo '=== procs ==='; pgrep -af 'train_remote_v4|chain_v1|chain_v2|expand_300m' | grep -v grep | head -5 || echo NONE; } 2>&1 | "
       "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=r6_final_004.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "r6-final-004", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

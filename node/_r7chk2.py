#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ echo '=== now ==='; date '+%H:%M:%S'; "
       "echo '=== corpus_R7 full ==='; ls -la /mnt/workspace/node1/chain/corpus_R7/; "
       "echo '=== corpus_reserve ==='; ls -la /mnt/workspace/node1/chain/corpus_reserve/ | head -8; "
       "echo '=== train alive ==='; pgrep -af train_remote_v4 | grep -v grep | wc -l; "
       "echo '=== run_tmp ==='; ls -la /mnt/workspace/node1/chain/run_tmp/ 2>/dev/null | head -5; "
       "echo '=== chain_v2.log tail 4 ==='; tail -4 /mnt/workspace/node1/chain/chain_v2.log; } 2>&1 | "
       "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=r7_chk_002.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "r7-chk-002", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ echo '=== chain.log tail 40 ==='; tail -40 /mnt/workspace/node1/chain/chain.log 2>/dev/null; "
       "echo '=== done_steps now ==='; cat /mnt/workspace/node1/chain/done_steps.txt 2>/dev/null; echo; "
       "echo '=== procs ==='; pgrep -af 'train_remote_v4|chain_v1|expand' | grep -v grep | head -4; "
       "echo '=== upload dir size ==='; du -sh /mnt/workspace/node1/chain/run_tmp/checkpoint-latest 2>/dev/null; "
       "echo '=== ECS upload test ==='; echo ok | curl -s -m20 -X POST 'http://47.99.131.12:8081/upload?name=up_test_003.txt&key=k0u4mail' --data-binary @-; echo; } 2>&1 | "
       "curl -s -m120 -X POST 'http://47.99.131.12:8081/upload?name=r6_final_001.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "r6-final-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ echo '=== chain.log tail 15 ==='; tail -15 /mnt/workspace/node1/chain/chain.log 2>/dev/null; "
       "echo '=== done_steps ==='; cat /mnt/workspace/node1/chain/done_steps.txt 2>/dev/null; echo; "
       "echo '=== procs ==='; pgrep -af 'train_remote_v4|chain_v1|expand|python3' | grep -v grep | head -5; "
       "echo '=== mid-size upload test 10MB ==='; head -c 10485760 /dev/urandom > /tmp/mid.bin; "
       "curl -s -m120 -X POST 'http://47.99.131.12:8081/upload?name=mid_test_010.bin&key=k0u4mail' --data-binary @/tmp/mid.bin; echo; "
       "echo '=== 100MB test ==='; head -c 104857600 /dev/urandom > /tmp/big.bin; "
       "curl -s -m300 -X POST 'http://47.99.131.12:8081/upload?name=big_test_100.bin&key=k0u4mail' --data-binary @/tmp/big.bin; echo; "
       "rm -f /tmp/mid.bin /tmp/big.bin; } 2>&1 | "
       "curl -s -m400 -X POST 'http://47.99.131.12:8081/upload?name=r6_final_002.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "r6-final-002", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

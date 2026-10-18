#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check corpus_04b MS download progress (correct log names)."""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
CMD = (
    "{ echo '--- ms dl log ---'; cat /mnt/workspace/node1/corpus_04b_ms.log 2>/dev/null; "
    "echo '--- corpus_04b tree ---'; du -sh /mnt/workspace/node1/corpus_04b/* 2>/dev/null; "
    "ls -la /mnt/workspace/node1/corpus_04b/ufw/ 2>/dev/null; "
    "echo '--- R6 ---'; tail -1 /mnt/workspace/node1/chain/chain.log 2>/dev/null; "
    "cat /mnt/workspace/node1/chain/done_steps.txt 2>/dev/null; "
    "} 2>&1 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=dl_prog_002.txt&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "dl-prog-002", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR:", e)

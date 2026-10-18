#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check corpus_04b download progress on node1."""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
CMD = (
    "{ echo '--- ufw dl log ---'; cat /mnt/workspace/node1/corpus_04b_dl.log 2>/dev/null; "
    "echo '--- l3 dl log ---'; cat /mnt/workspace/node1/corpus_04b_l3.log 2>/dev/null; "
    "echo '--- corpus_04b tree ---'; du -sh /mnt/workspace/node1/corpus_04b/* 2>/dev/null; "
    "echo '--- R6 progress ---'; tail -2 /mnt/workspace/node1/chain/chain.log 2>/dev/null; "
    "cat /mnt/workspace/node1/chain/done_steps.txt 2>/dev/null; "
    "} 2>&1 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=dl_prog_001.txt&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "dl-prog-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR:", e)

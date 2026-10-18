#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ echo '--- R6 tail ---'; tail -3 /mnt/workspace/node1/chain/chain.log 2>/dev/null; "
       "echo '--- done_steps ---'; cat /mnt/workspace/node1/chain/done_steps.txt 2>/dev/null; "
       "echo '--- heartbeat ---'; cat /mnt/workspace/node1/chain/heartbeat 2>/dev/null; "
       "echo '--- l3 local ---'; } 2>&1 | "
       "curl -s -X POST 'http://47.99.131.12:8081/upload?name=r6_chk_001.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "r6-chk-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST ->", r.status)
except Exception as e:
    print("ERR:", e)

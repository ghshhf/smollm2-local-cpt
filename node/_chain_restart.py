#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("D=/mnt/workspace/node1/chain; "
       "rm -f $D/chain.pid; "
       "(nohup sh $D/chain_v2.sh >/dev/null 2>&1 </dev/null &); "
       "sleep 8; "
       "{ echo '=== now ==='; date '+%H:%M:%S'; "
       "echo '=== procs ==='; pgrep -af 'chain_v2|train_remote_v4' | grep -v grep | head -3; "
       "echo '=== chain_v2.log tail 10 ==='; tail -10 $D/chain_v2.log 2>/dev/null; } 2>&1 | "
       "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=chain_restart_001.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "chain-restart-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

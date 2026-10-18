#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
B64 = open("D:/work/_local_ai/train_15b/node/train_live_push.b64").read().strip()
CMD = ("echo '%s' | base64 -d > /mnt/workspace/node1/train_live_push.py; "
       "if pgrep -f train_live_pus[h] >/dev/null; then echo ALREADY_RUNNING; "
       "else (nohup python3 -u /mnt/workspace/node1/train_live_push.py "
       "> /mnt/workspace/node1/train_live.log 2>&1 </dev/null &); echo STARTED; fi; "
       "sleep 3; pgrep -af train_live_pus[h] | head -2; "
       "{ echo '--- first log ---'; tail -3 /mnt/workspace/node1/train_live.log 2>/dev/null; } 2>&1 | "
       "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=live_dispatch_001.txt&key=k0u4mail' --data-binary @-") % B64
body = json.dumps({"cmd": CMD, "id": "live-dispatch-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ echo '=== now ==='; date '+%H:%M:%S'; echo '=== expand_run.log ==='; cat /mnt/workspace/node1/chain/expand_run.log 2>/dev/null | grep -vE 'Loading|torch_dtype|warnings.warn' | tail -10; "
       "echo '=== ckpt config ==='; python3 -c \"import json;c=json.load(open('/mnt/workspace/node1/chain/ckpt/config.json'));print('inter',c['intermediate_size'],'layers',c['num_hidden_layers'])\"; "
       "echo '=== ckpt files ==='; ls -la /mnt/workspace/node1/chain/ckpt/ | head -8; "
       "echo '=== chain_v2.log tail 12 ==='; tail -12 /mnt/workspace/node1/chain/chain_v2.log 2>/dev/null; "
       "echo '=== procs ==='; pgrep -af 'chain_v2|train_remote_v4' | grep -v grep | head -3; } 2>&1 | "
       "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=expand_chk_001.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "expand-chk-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

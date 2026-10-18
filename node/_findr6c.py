#!/usr/bin/env python3
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = r"""{ echo '=== chain dir ==='; ls -la /mnt/workspace/node1/chain/ | grep -vE '^total';
echo '=== run_tmp ==='; ls -la /mnt/workspace/node1/chain/run_tmp/ 2>/dev/null | head -8;
echo '=== ckpt ==='; ls -la /mnt/workspace/node1/chain/ckpt/ 2>/dev/null | head -6;
echo '=== done ==='; cat /mnt/workspace/node1/chain/done_steps.txt; echo; } 2>&1 | curl -s -m45 -X POST 'http://47.99.131.12:8081/upload?name=find_r6_003.txt&key=k0u4mail' --data-binary @-"""
body = json.dumps({"cmd": CMD, "id": "find-r6-003", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

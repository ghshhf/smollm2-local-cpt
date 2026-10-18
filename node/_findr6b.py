#!/usr/bin/env python3
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = r"""{ echo '=== all safetensors ==='; find /mnt/workspace/node1/chain -name '*.safetensors' -size +100M | while read f; do ls -la "$f"; done;
echo '=== chain dir ==='; ls -la /mnt/workspace/node1/chain/;
echo '=== run_tmp ==='; ls -la /mnt/workspace/node1/chain/run_tmp/ 2>/dev/null | head -10;
echo '=== sha256 ==='; find /mnt/workspace/node1/chain -name 'model.safetensors' -size +500M | while read f; do echo "$f $(stat -c%s $f) $(sha256sum $f | cut -c1-16)"; done;
echo '=== done_steps ==='; cat /mnt/workspace/node1/chain/done_steps.txt; echo;
echo '=== R7 train alive ==='; pgrep -c -f train_remote_v4; } 2>&1 | curl -s -m90 -X POST 'http://47.99.131.12:8081/upload?name=find_r6_002.txt&key=k0u4mail' --data-binary @-"""
body = json.dumps({"cmd": CMD, "id": "find-r6-002", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

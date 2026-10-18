#!/usr/bin/env python3
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ echo '=== all model.safetensors ==='; find /mnt/workspace/node1/chain -name 'model.safetensors' -exec ls -la {} \; 2>/dev/null; "
       "echo '=== promote/backup dirs ==='; ls -la /mnt/workspace/node1/chain/ ; "
       "echo '=== run_tmp ==='; ls -la /mnt/workspace/node1/chain/run_tmp/ 2>/dev/null | head -10; "
       "echo '=== sha256 of big ones ==='; for f in $(find /mnt/workspace/node1/chain -name 'model.safetensors' -size +500M 2>/dev/null); do echo \\\"$f $(stat -c%s $f) $(sha256sum $f | cut -c1-16)\\\"; done; "
       "echo '=== done_steps ==='; cat /mnt/workspace/node1/chain/done_steps.txt; echo; "
       "echo '=== R7 alive ==='; pgrep -af train_remote_v4 | grep -v grep | wc -l; } 2>&1 | "
       "curl -s -m90 -X POST 'http://47.99.131.12:8081/upload?name=find_r6_001.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "find-r6-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

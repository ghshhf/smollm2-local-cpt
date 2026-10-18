#!/usr/bin/env python3
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = r"""{ echo '=== pull2 nohup 日志 ==='; cat /tmp/r6final_pull.log 2>&1 | head -10;
echo '=== pull1 日志 ==='; cat /tmp/r6final/pull.log 2>&1 | head -5;
echo '=== /tmp 下 r6 相关 ==='; ls -la /tmp/ | grep -i r6;
echo '=== node_v5 自己的日志尾 ==='; tail -5 /mnt/workspace/node1/node_v5.log 2>/dev/null || find /mnt/workspace/node1 -name 'node_v5*.log' -exec tail -5 {} \; 2>/dev/null; } 2>&1 | curl -s -m45 -X POST 'http://47.99.131.12:8081/upload?name=r6f_logread_001.txt&key=k0u4mail' --data-binary @-"""
body = json.dumps({"cmd": CMD, "id": "r6-logread-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

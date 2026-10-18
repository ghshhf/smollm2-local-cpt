#!/usr/bin/env python3
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ df -h / 2>/dev/null | tail -2; df -h /tmp 2>/dev/null | tail -1; "
       "ls -la /tmp/up_test_*.bin 2>/dev/null | head -5; "
       "du -sh $(dirname $(find / -maxdepth 3 -name 'ecs*' -type d 2>/dev/null | head -1)) 2>/dev/null | head -2; } 2>&1 | "
       "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=ecs_disk_001.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "ecs-disk-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

#!/usr/bin/env python3
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = r"""{ ls /tmp/r6final/ 2>&1 | head -8; echo ---; ls -la /tmp/r6final/model.safetensors.part* 2>/dev/null | wc -l; } 2>&1 | curl -s -m45 -X POST 'http://47.99.131.12:8081/upload?name=r6f_check_001.txt&key=k0u4mail' --data-binary @-"""
body = json.dumps({"cmd": CMD, "id": "r6-check-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

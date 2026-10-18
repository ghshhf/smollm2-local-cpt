#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""List Ultra-FineWeb-zh file tree + sizes on hf-mirror via ECS /task."""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
CMD = (
    "{ echo '=== ufw zh tree ==='; "
    "curl -s -m 60 'https://hf-mirror.com/api/datasets/openbmb/Ultra-FineWeb/tree/main/data/ultrafineweb_zh' | head -c 4000; echo; "
    "echo '=== ufw zh recursive ==='; "
    "curl -s -m 60 'https://hf-mirror.com/api/datasets/openbmb/Ultra-FineWeb/tree/main/data/ultrafineweb_zh?recursive=true' | head -c 4000; echo; "
    "echo '=== modelscope ufw ==='; "
    "curl -s -m 60 'https://modelscope.cn/api/v1/datasets/AI-ModelScope/Ultra-FineWeb/repo/files?Revision=master&Root=data/ultrafineweb_zh' | head -c 3000; echo; "
    "} 2>&1 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=ufw_list_001.txt&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "ufw-list-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR:", e)

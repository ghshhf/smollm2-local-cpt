#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_setup.py — 探 node1: 仓库/git/远程 / 0.1B 分片 / 重组脚本 / HF&ModelScope 连通 / 磁盘 / python。
结果回传 ECS /upload，本机 curl /file/<name> 取回。
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
ts = int(time.time())
name = f"probe_setup_{ts}.txt"

CMD = (
    "REPO=$(ls -d /mnt/workspace/train_15b 2>/dev/null | head -1); "
    "{ "
    "echo '=== REPO ==='; echo \"REPO=$REPO\"; "
    "echo '=== git remote ==='; [ -n \"$REPO\" ] && cd \"$REPO\" 2>/dev/null && (git remote -v 2>&1 | head -4; git log --oneline -1 2>&1 | head -1) || echo NO_REPO; "
    "echo '=== 0.1B dir on node1 ==='; find /mnt/workspace -maxdepth 5 -type d -name '0.1B' 2>/dev/null | head; "
    "echo '=== reassemble scripts ==='; find /mnt/workspace -maxdepth 6 -name 'reassemble*.py' 2>/dev/null | head; "
    "echo '=== HF http ==='; curl -s -m 8 -o /dev/null -w '%{http_code}' https://huggingface.co 2>&1; echo; "
    "echo '=== ModelScope http ==='; curl -s -m 8 -o /dev/null -w '%{http_code}' https://modelscope.cn 2>&1; echo; "
    "echo '=== disk /mnt/workspace ==='; df -h /mnt/workspace 2>/dev/null | tail -1; "
    "echo '=== python ==='; python3 --version 2>&1; "
    "} 2>&1 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=" + name + "&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": f"probe-setup-{ts}", "ts": ts}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
    print("fetch:", f"{ECS}/file/{name}")
except Exception as e:
    print("POST ERR:", e)

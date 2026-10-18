#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_r7_stall.py — 诊断 node1 R7 阶段停滞（心跳停在 10-17 19:24）。"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
NODE = "/mnt/workspace/node1"

CMD = (
    "{ echo '=== TIME ==='; date '+%%Y-%%m-%%d %%H:%%M:%%S'; "
    "echo '=== PROCS(train_remote/train_*.py/chain) ==='; ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | grep -E 'train_remote|train_|chain_|python3' | grep -v grep | head -12; "
    "echo '=== BASE_TOK ==='; ls -la %s/chain/base_tok/ 2>/dev/null; "
    "echo '=== CHAIN_DIR ==='; ls -lt %s/chain/ 2>/dev/null | head -15; "
    "echo '=== CHAINV2_LOG ==='; ls -lt %s/*v2*.log %s/chain/*v2*.log 2>/dev/null | head -5; tail -25 %s/chain/chain_v2.log 2>/dev/null; "
    "echo '=== RECENT_FILES ==='; find %s -maxdepth 3 -newermt '-6 hours' -type f 2>/dev/null | head -20; "
    "echo '=== DISK ==='; df -h %s 2>/dev/null | tail -2; "
    "echo '=== LOAD ==='; uptime; } 2>&1 | "
    "curl -s -m90 -X POST 'http://47.99.131.12:8081/upload?name=diag_r7_stall.txt&key=k0u4mail' --data-binary @-"
) % (NODE, NODE, NODE, NODE, NODE, NODE, NODE)

body = json.dumps({"cmd": CMD, "id": "diag-r7-stall-%d" % int(time.time()), "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
print("POST ->", urllib.request.urlopen(req, timeout=20).read().decode()[:200])

txt = ""
for _ in range(20):
    time.sleep(5)
    try:
        txt = urllib.request.urlopen(
            "%s/file/diag_r7_stall.txt?key=%s" % (ECS, KEY), timeout=40
        ).read().decode("utf-8", "replace")
    except Exception:
        txt = ""
    if "=== LOAD ===" in txt:
        break
print(txt if txt.strip() else "(无回传)")

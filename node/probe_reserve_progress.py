#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_reserve_progress.py — 查 node1 上 build_reserve 进度 / 卡点
结果上报 ECS /file/status_reserve_probe.txt
"""
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = (
    "{ "
    "echo '=== build_reserve.py running? ==='; ps aux | grep -c '[b]uild_reserve.py'; "
    "echo '=== /tmp/reserve ==='; ls -la /tmp/reserve 2>/dev/null | tail -n8; "
    "echo '=== Classical-Modern clone size ==='; du -sh /tmp/Classical-Modern 2>/dev/null; "
    "echo '=== reserve dir ==='; ls -la /mnt/workspace/node1/chain/corpus_reserve 2>/dev/null; "
    "echo '=== last lines of reserve.log ==='; tail -n 15 /tmp/reserve.log 2>/dev/null; "
    "} | curl -s -X POST 'http://47.99.131.12:8081/upload?name=status_reserve_probe.txt&key=k0u4mail' --data-binary @-"
)
body = json.dumps({"cmd": CMD, "id": "reserve-probe", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:120])
except Exception as e:
    print("POST ERR:", e)

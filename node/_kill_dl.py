#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kill stuck hf-mirror download processes on node1 (xet CDN unreachable)."""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
CMD = (
    "pkill -f 'dl_[u]fw.sh'; pkill -f 'dl_[l]3.sh'; "
    "pkill -f 'ultrafine[w]eb'; pkill -f 'hf-mirror.[c]om'; "
    "sleep 1; "
    "{ echo 'killed at $(date)'; ps -eo args 2>/dev/null | grep -v grep | grep -cE 'ultrafine|hf-mirror' || true; "
    "du -sh /mnt/workspace/node1/corpus_04b/* 2>/dev/null; } 2>&1 | "
    "curl -s -X POST 'http://47.99.131.12:8081/upload?name=kill_dl_001.txt&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "kill-dl-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR:", e)

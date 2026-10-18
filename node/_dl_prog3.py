#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snapshot UFW conversion result + dispatch poetry clone."""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
CMD = (
    "{ echo '--- ms dl log tail ---'; tail -25 /mnt/workspace/node1/corpus_04b_ms.log 2>/dev/null; "
    "echo '--- corpus_04b ---'; du -sh /mnt/workspace/node1/corpus_04b/* 2>/dev/null; "
    "echo '--- R6 ---'; tail -1 /mnt/workspace/node1/chain/chain.log 2>/dev/null; "
    "} 2>&1 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=dl_prog_003.txt&key=k0u4mail' --data-binary @-"
)
body = json.dumps({"cmd": CMD, "id": "dl-prog-003", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("prog POST ->", r.status)
except Exception as e:
    print("prog ERR:", e)

# git clone poetry (no LFS needed for small repo)
CMD2 = (
    "cd /mnt/workspace/node1/corpus_04b && "
    "(nohup bash -c 'git clone --depth 1 https://www.modelscope.cn/datasets/modelscope/chinese-poetry-collection.git poetry 2>&1; "
    "du -sh poetry; find poetry -name \"*.jsonl\" -o -name \"*.json\" | head -5; echo POETRY_CLONE_DONE' "
    "> /mnt/workspace/node1/poetry_clone.log 2>&1 </dev/null &) && echo CLONE_STARTED"
)
body2 = json.dumps({"cmd": CMD2, "id": "poetry-clone-001", "ts": int(time.time())}).encode()
req2 = urllib.request.Request(ECS + "/task", data=body2,
                              headers={"Content-Type": "application/json"}, method="POST")
try:
    r2 = urllib.request.urlopen(req2, timeout=15)
    print("clone POST ->", r2.status)
except Exception as e:
    print("clone ERR:", e)

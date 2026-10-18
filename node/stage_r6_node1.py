#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage_r6_node1.py — 让 node1 从 ECS 拉回 r6_00..03 并拼成 $W/corpus_r6/zh_text_r6.jsonl (R6 语料上 NAS)
结果上报到 ECS /file/status_r6stage.txt
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
CMD = ("mkdir -p /mnt/workspace/node1/chain/corpus_r6; "
       "rm -f /mnt/workspace/node1/chain/corpus_r6/zh_text_r6.jsonl; "
       "for i in 00 01 02 03; do "
       "curl -s -m300 -o /tmp/r6_$i http://47.99.131.12:8081/file/r6_$i && "
       "cat /tmp/r6_$i >> /mnt/workspace/node1/chain/corpus_r6/zh_text_r6.jsonl; done; "
       "{ echo LINES=$(wc -l < /mnt/workspace/node1/chain/corpus_r6/zh_text_r6.jsonl); "
       "echo SIZE=$(wc -c < /mnt/workspace/node1/chain/corpus_r6/zh_text_r6.jsonl); "
       "} | curl -s -X POST 'http://47.99.131.12:8081/upload?name=status_r6stage.txt&key=k0u4mail' --data-binary @-")

body = json.dumps({"cmd": CMD, "id": "stage-r6", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR (task may still be queued):", e)

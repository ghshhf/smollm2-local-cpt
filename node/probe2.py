#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe2.py — 全状态探针: done_steps / 当前进程 / config(intermediate,layers) / corpus_reserve 清单 / 磁盘"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
W = "/mnt/workspace/node1/chain"
CMD = (
    "{ echo NOW; date '+%m-%d %H:%M:%S'; "
    "echo DONESTEPS; cat " + W + "/done_steps.txt 2>/dev/null; "
    "echo PROC; ps aux|grep -E 'train_remote|chain_v1|chain_v2|expand'|grep -v grep|head; "
    "echo CONFIG; python3 -c \"import json;c=json.load(open('" + W + "/ckpt/config.json'));"
    "print('inter=',c.get('intermediate_size'),'layers=',c.get('num_hidden_layers'),'hidden=',c.get('hidden_size'))\" 2>/dev/null; "
    "echo RESERVE; ls -la " + W + "/corpus_reserve/ 2>/dev/null; "
    "echo DISK; df -h /mnt/workspace 2>/dev/null | tail -1; "
    "} | curl -s -X POST '" + ECS + "/upload?name=status_probe2.txt&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "probe2", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR:", e)

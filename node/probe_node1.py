#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_node1.py — 向 ECS /task 派一个状态探针, node1 跑完后把结果上传到 ECS /file/status_probe.txt"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
CMD = ("{ echo NOW; date '+%m-%d %H:%M:%S'; echo DONESTEPS; cat /mnt/workspace/node1/chain/done_steps.txt 2>/dev/null; "
       "echo PROCS; ps aux|grep -E 'train_remote|chain_v1'|grep -v grep; "
       "echo CHAINLINES; wc -l /mnt/workspace/node1/chain/chain_v1.sh; "
       "echo HASR6; grep -c 'R6 段' /mnt/workspace/node1/chain/chain_v1.sh; "
       "echo CORPUSR6; ls -la /mnt/workspace/node1/chain/corpus_r6/ 2>/dev/null; "
       "} | curl -s -X POST 'http://47.99.131.12:8081/upload?name=status_probe.txt&key=k0u4mail' --data-binary @-")

body = json.dumps({"cmd": CMD, "id": "probe-r6", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR:", e)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_dup.py — 精确查是否有重复的 chain / 训练进程在并发跑(会导致 ckpt 竞态)。"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
W = "/mnt/workspace/node1/chain"
NAME = sys.argv[1] if len(sys.argv) > 1 else "dup_probe1.txt"

CMD = (
    "cd " + W + "; "
    "{ echo TIME; date '+%m-%d %H:%M:%S'; "
    "echo DONE; cat done_steps.txt 2>/dev/null; "
    "echo '--- CHAIN 进程(应只有1个) ---'; "
    "ps -eo pid,lstart,etime,cmd | grep '[c]hain_v1.sh'; "
    "echo '--- TRAIN 主进程(应只有1组) ---'; "
    "ps -eo pid,lstart,etime,cmd | grep '[t]rain_remote_v4' | grep -v 'grep'; "
    "echo '--- run_tmp ---'; ls -la run_tmp/checkpoint-latest/ 2>/dev/null | head -5; "
    "echo '--- TAIL ---'; tail -4 chain.log; "
    "} 2>&1 | curl -s -X POST '" + ECS + "/upload?name=" + NAME + "&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "dup-" + NAME.replace(".txt", ""),
                   "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
except Exception as e:
    print("POST ERR:", e)

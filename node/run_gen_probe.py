#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_gen_probe.py — 派 node1 跑一次生成抽样, 结果写 gen_probe<N>.txt(文件名轮换, 避 ECS 文件锁)。"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
W = "/mnt/workspace/node1/chain"
NAME = sys.argv[1] if len(sys.argv) > 1 else "gen_probe1.txt"

CMD = (
    "curl -s -m120 -o " + W + "/node1_gen.py " + ECS + "/file/node1_gen.py; "
    "cd " + W + " && python3 node1_gen.py "
    "--base " + W + "/ckpt --tok " + W + "/base_tok --max_new 70 2>&1 "
    "| curl -s -X POST '" + ECS + "/upload?name=" + NAME + "&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "gen-" + NAME.replace(".txt", ""),
                   "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
except Exception as e:
    print("POST ERR:", e)

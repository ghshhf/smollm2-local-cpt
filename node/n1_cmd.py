#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n1_cmd.py — 通用派活: 把一段 shell 命令交给 node1 执行。
用法: python3 n1_cmd.py "shell 命令" [任务id]
node1 的 agent 轮询 ECS /task 取走执行; 结果如需回传, 命令里自己 curl 上传(文件名务必轮换)。
"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"

cmd = sys.argv[1]
tid = sys.argv[2] if len(sys.argv) > 2 else "cmd-%d" % int(time.time())

body = json.dumps({"cmd": cmd, "id": tid, "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| id =", tid)
except Exception as e:
    print("POST ERR:", e)

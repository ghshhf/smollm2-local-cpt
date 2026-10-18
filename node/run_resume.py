#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_resume.py — 确认链已停机 -> 清 stop 标志 -> 后台重启 chain_v1.sh(用清洗后语料进 R5)。
结果写 resume<N>.txt。"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
W = "/mnt/workspace/node1/chain"
NAME = sys.argv[1] if len(sys.argv) > 1 else "resume1.txt"

CMD = (
    "cd " + W + "; "
    "{ echo TIME; date '+%m-%d %H:%M:%S'; "
    "echo DONE_BEFORE; cat done_steps.txt 2>/dev/null; "
    "echo STOP_BEFORE; if [ -f stop ]; then echo present; else echo absent; fi; "
    "echo PROC_BEFORE; ps aux|grep -E '[c]hain_v1|[t]rain_remote_v4' | wc -l; "
    "rm -f stop; "
    "(nohup sh chain_v1.sh >> " + W + "/chain.log 2>&1 < /dev/null &); "
    "sleep 8; "
    "echo PROC_AFTER; ps aux|grep -E '[c]hain_v1' | wc -l; "
    "echo TAIL; tail -5 " + W + "/chain.log; "
    "} 2>&1 | curl -s -X POST '" + ECS + "/upload?name=" + NAME + "&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "resume-" + NAME.replace(".txt", ""),
                   "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
except Exception as e:
    print("POST ERR:", e)

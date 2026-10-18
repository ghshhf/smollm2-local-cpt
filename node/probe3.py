#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe3.py — 状态探针(写新文件名 status_probe3.txt, 规避 ECS 同名文件锁)。
含: done_steps / 当前进程 / 训练日志尾 / config / 各段语料进度。"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
W = "/mnt/workspace/node1/chain"
CMD = (
    "{ echo NOW; date '+%m-%d %H:%M:%S'; "
    "echo DONESTEPS; cat " + W + "/done_steps.txt 2>/dev/null; "
    "echo LASTLOG; tail -12 " + W + "/chain.log 2>/dev/null; "
    "echo TRAINLOG; tail -8 " + W + "/chain.log 2>/dev/null | grep -E 'step|chunk' | tail -4; "
    "echo PROC; ps aux|grep -E '[t]rain_remote_v4|[c]hain_v1' | awk '{print $2,$10,$11,$12,$13}'; "
    "echo CONFIG; python3 -c \"import json;c=json.load(open('"+W+"/ckpt/config.json'));"
    "print('inter=',c.get('intermediate_size'),'layers=',c.get('num_hidden_layers'))\" 2>/dev/null; "
    "} | curl -s -X POST '" + ECS + "/upload?name=status_probe3.txt&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "probe3", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR:", e)

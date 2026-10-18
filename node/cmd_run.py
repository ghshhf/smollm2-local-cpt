#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cmd_run.py — 通用: 经信箱派任意 shell 命令并把输出回传本地。
用法: python cmd_run.py "shell 命令"         (命令里不要出现 % 和双引号)
"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
cmd = sys.argv[1] if len(sys.argv) > 1 else "uptime"
name = sys.argv[2] if len(sys.argv) > 2 else "cmd_run.txt"

FULL = cmd + " 2>&1 | curl -s -m60 -X POST '" + ECS + "/upload?name=" + name + "&key=" + KEY + "' --data-binary @-"
body = json.dumps({"cmd": FULL, "id": "cmd-%d" % int(time.time()), "ts": int(time.time())}).encode()
urllib.request.urlopen(urllib.request.Request(ECS + "/task", data=body,
                       headers={"Content-Type": "application/json"}, method="POST"), timeout=20).read()

txt = ""
for _ in range(16):
    time.sleep(4)
    try:
        txt = urllib.request.urlopen("%s/file/%s?key=%s" % (ECS, name, KEY), timeout=30).read().decode("utf-8", "replace")
    except Exception:
        txt = ""
    if txt.strip():
        break
print(txt if txt.strip() else "(无回传)")

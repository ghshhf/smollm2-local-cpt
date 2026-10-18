#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch_remote.py — 经信箱把远程文件(cat)回传到本地查看。用法: python fetch_remote.py <远端路径> [回传名]"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/workspace/node1/chain/expand_300m_node1.py"
name = sys.argv[2] if len(sys.argv) > 2 else "fetched.txt"

CMD = ("cat " + path + " 2>&1 | curl -s -m60 -X POST '" + ECS + "/upload?name=" + name
       + "&key=" + KEY + "' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "fetch-%d" % int(time.time()), "ts": int(time.time())}).encode()
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

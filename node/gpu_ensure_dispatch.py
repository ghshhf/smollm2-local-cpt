#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_ensure_dispatch.py — 经 ECS 信箱直接派 gpu_ensure 到远程(无需人工粘命令)。

注意: node1(CPU) 与 GPU 实例同时跑 node_v5 领活, /task 落点随机。
      故命令内先自检 torch.cuda.is_available(): 只有 GPU 实例才真正执行 ensure,
      两台都回传 host + cuda 状态, 便于判断落点。
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"

CMD = (
    "{ echo HOST=$(hostname); "
    "python3 -c 'import torch;print(\"CUDA\", torch.cuda.is_available())' 2>&1 | head -2; "
    "if python3 -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)'; then "
    "echo '--- 本机有 GPU, 执行 ensure ---'; "
    "curl -s -o /tmp/e.py 'http://47.99.131.12:8081/file/gpu_ensure.py?key=%s'; "
    "python3 /tmp/e.py; "
    "else echo '--- 本机无 GPU, 跳过 ---'; fi; } 2>&1 | "
    "curl -s -m90 -X POST 'http://47.99.131.12:8081/upload?name=gpu_ensure_dispatch.txt&key=%s' --data-binary @-"
) % (KEY, KEY)

body = json.dumps({"cmd": CMD, "id": "ensure-%d" % int(time.time()), "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
print("POST ->", urllib.request.urlopen(req, timeout=20).read().decode()[:200])

txt = ""
for _ in range(24):
    time.sleep(5)
    try:
        txt = urllib.request.urlopen(
            "%s/file/gpu_ensure_dispatch.txt?key=%s" % (ECS, KEY), timeout=40
        ).read().decode("utf-8", "replace")
    except Exception:
        txt = ""
    if "DONE" in txt or "跳过" in txt or "无 GPU" in txt:
        break
print(txt if txt.strip() else "(无回传)")

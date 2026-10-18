#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_probe.py — 经信箱快速探活 GPU 训练(无长 sleep, 避免执行器超时)。
用法: python gpu_probe.py        # 可反复执行
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"

CMD = (
    "{ echo HOST=$(hostname); date '+%%H:%%M:%%S'; "
    "echo '--- GPU ---'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1; "
    "echo '--- PROC ---'; ps -eo pid,etime,pcpu,args | grep -F 'gpu_autopilot' | grep -v grep | head -3; "
    "ps -eo pid,etime,pcpu,args | grep -F 'train_remote_v5_gpu' | grep -v grep | head -2; "
    "echo '--- CKPT ---'; ls -lt /mnt/workspace/node1/chain/ckpt_gpu/ 2>/dev/null | head -4; "
    "cat /mnt/workspace/node1/chain/ckpt_gpu/checkpoint-latest/step.txt 2>/dev/null; "
    "echo '--- LOG ---'; tail -10 /root/autopilot.log 2>/dev/null; } 2>&1 | "
    "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=gpu_probe.txt&key=%s' --data-binary @-"
) % KEY

body = json.dumps({"cmd": CMD, "id": "probe-%d" % int(time.time()), "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
urllib.request.urlopen(req, timeout=20).read()

txt = ""
for _ in range(20):
    time.sleep(4)
    try:
        txt = urllib.request.urlopen(
            "%s/file/gpu_probe.txt?key=%s" % (ECS, KEY), timeout=30
        ).read().decode("utf-8", "replace")
    except Exception:
        txt = ""
    if "--- LOG ---" in txt:
        break
print(txt if txt.strip() else "(无回传)")

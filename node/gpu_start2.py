#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_start2.py — 强启动 autopilot v2: 去掉 pkill(避免自杀), setsid nohup 守护。
已确认无残留进程, 无需清理。短 sleep(12s) 避免执行器超时。
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"

CMD = (
    "{ echo HOST=$(hostname); date '+%%H:%%M:%%S'; "
    "if python3 -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)'; then "
    "curl -s -o /root/gpu_autopilot.py 'http://47.99.131.12:8081/file/gpu_autopilot.py?key=%s'; "
    "wc -c /root/gpu_autopilot.py; "
    "cd /root && setsid nohup python3 -u /root/gpu_autopilot.py > /root/autopilot.log 2>&1 < /dev/null & "
    "sleep 14; "
    "echo '--- LOG ---'; tail -8 /root/autopilot.log 2>/dev/null; "
    "echo '--- GPU ---'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; "
    "else echo '--- 无 GPU, 跳过 ---'; fi; } 2>&1 | "
    "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=gpu_start2.txt&key=%s' --data-binary @-"
) % (KEY, KEY)

body = json.dumps({"cmd": CMD, "id": "start2-%d" % int(time.time()), "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
urllib.request.urlopen(req, timeout=20).read()

txt = ""
for _ in range(20):
    time.sleep(4)
    try:
        txt = urllib.request.urlopen(
            "%s/file/gpu_start2.txt?key=%s" % (ECS, KEY), timeout=30
        ).read().decode("utf-8", "replace")
    except Exception:
        txt = ""
    if "--- GPU ---" in txt or "跳过" in txt:
        break
print(txt if txt.strip() else "(无回传)")

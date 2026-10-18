#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_start_now.py — 经信箱强启动 autopilot（setsid nohup 守护，抗 Jupyter 断连）。

只在 torch.cuda.is_available() 的实例上真正启动; 另一台只回传状态。
先 pkill 同名旧进程(用字符类规避自杀), 再 nohup 拉起。
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"

CMD = (
    "{ echo HOST=$(hostname); "
    "if python3 -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)'; then "
    "echo '--- 有 GPU, 强启动 ---'; "
    "pkill -f '[g]pu_autopilot' 2>/dev/null; sleep 2; "
    "curl -s -o /root/gpu_autopilot.py 'http://47.99.131.12:8081/file/gpu_autopilot.py?key=%s'; "
    "wc -c /root/gpu_autopilot.py; "
    "cd /root && setsid nohup python3 -u /root/gpu_autopilot.py > /root/autopilot.log 2>&1 < /dev/null & "
    "sleep 45; "
    "echo '--- ps ---'; ps -eo pid,etime,pcpu,args | grep -E 'gpu_autopilot|train_remote_v5_gpu' | grep -v grep | head -4; "
    "echo '--- nvidia ---'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; "
    "echo '--- log ---'; tail -14 /root/autopilot.log 2>/dev/null; "
    "else echo '--- 无 GPU, 跳过 ---'; fi; } 2>&1 | "
    "curl -s -m120 -X POST 'http://47.99.131.12:8081/upload?name=gpu_start_now.txt&key=%s' --data-binary @-"
) % (KEY, KEY)

body = json.dumps({"cmd": CMD, "id": "start-%d" % int(time.time()), "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
print("POST ->", urllib.request.urlopen(req, timeout=20).read().decode()[:200])

txt = ""
for _ in range(30):
    time.sleep(5)
    try:
        txt = urllib.request.urlopen(
            "%s/file/gpu_start_now.txt?key=%s" % (ECS, KEY), timeout=40
        ).read().decode("utf-8", "replace")
    except Exception:
        txt = ""
    if "nvidia" in txt or "跳过" in txt:
        break
print(txt if txt.strip() else "(无回传)")

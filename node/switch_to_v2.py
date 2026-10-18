#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""switch_to_v2.py — 延时自动切换: 让 v1 再跑 N 秒 -> 刹车 -> 启 autopilot2(从 R8 起)。

后台 setsid nohup 执行, 本命令立即返回(避免执行器超时)。
默认 1500s(25分钟, 约 R7 吃满 1 遍)后刹车; 90s 后写 stage.txt=R7 并启 v2。
v2 预算设 28h(35h44m 额度 - v1 已用 - 安全余量)。
"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
CH = "/mnt/workspace/node1/chain"
WAIT = sys.argv[1] if len(sys.argv) > 1 else "1500"

BG = (
    "sleep " + WAIT + "; "
    "touch " + CH + "/STOP_GPU; "
    "sleep 90; "
    "echo R7 > " + CH + "/ckpt_gpu/stage.txt; "
    "touch " + CH + "/gpu_hours_28; "
    "curl -s -o /root/gpu_autopilot2.py '" + ECS + "/file/gpu_autopilot2.py?key=" + KEY + "'; "
    "cd /root && python3 -u /root/gpu_autopilot2.py > /root/autopilot2.log 2>&1"
)

CMD = ("setsid nohup bash -c \"" + BG.replace('"', '\\\\"') + "\" > /root/switch.log 2>&1 < /dev/null & "
       "sleep 3; echo scheduled; ps -eo pid,args | grep -F 'sleep " + WAIT + "' | grep -v grep | head -2")
body = json.dumps({"cmd": CMD, "id": "switch-%d" % int(time.time()), "ts": int(time.time())}).encode()
urllib.request.urlopen(urllib.request.Request(ECS + "/task", data=body,
                       headers={"Content-Type": "application/json"}, method="POST"), timeout=20).read()

txt = ""
for _ in range(12):
    time.sleep(4)
    try:
        txt = urllib.request.urlopen("%s/file/switch_report.txt?key=%s" % (ECS, KEY), timeout=25).read().decode("utf-8", "replace")
    except Exception:
        txt = ""
    if "scheduled" in txt:
        break
print(txt if txt.strip() else "(派发已发出, 无即时回传)")

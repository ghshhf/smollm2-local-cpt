#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_watch_switch.py — 按「真实消费 token」触发 R7->R8 切换(严格判据, 非计时)。

在 GPU 实例后台跑:
  setsid nohup python3 -u /root/gpu_watch_switch.py > /root/watch_switch.log 2>&1 < /dev/null &

逻辑: 每 60s 解析 /root/autopilot.log 的「累计token=X M」,
      达到 R7 语料总量(默认 24.99M, 即 2499万) 即:
        1) touch STOP_GPU        (autopilot v1 在 chunk 边界优雅停, 不丢权重)
        2) 写 ckpt_gpu/stage.txt = R7  (v2 断点从 R8 起, 不重复吃)
        3) 等 90s 让 v1 收尾 + 存盘
        4) 若 v2 未运行(pgrep 带字符类锁, 防自匹配) 则启动 v2
      与「时间触发」并存: 两边启动 v2 前都检查 pgrep => 只有一个真正启动,
      不会双进程抢显存(各 15G > 24G 会 OOM)。
"""
import os, re, subprocess, time

LOG = "/root/autopilot.log"
STOP = "/mnt/workspace/node1/chain/STOP_GPU"
STAGE = "/mnt/workspace/node1/chain/ckpt_gpu/stage.txt"
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
THRESH = 24.99  # M token (R7 语料 2499万)
T0 = time.time()


def sh(c, timeout=300):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


log(f"watch start, threshold={THRESH}M token")
while time.time() - T0 < 6 * 3600:
    if os.path.exists("/root/gpu_autopilot2.py") is False:
        sh(f"curl -s -o /root/gpu_autopilot2.py '{ECS}/file/gpu_autopilot2.py?key={KEY}'")
    try:
        txt = open(LOG, encoding="utf-8", errors="ignore").read()
    except Exception:
        txt = ""
    ms = re.findall(r"累计token=([\d.]+)M", txt)
    tot = float(ms[-1]) if ms else 0.0
    log(f"累计 token = {tot}M / {THRESH}M")
    if tot >= THRESH:
        log("达到阈值 -> 刹车并切换 v2")
        sh(f"touch {STOP}")
        sh(f"echo R7 > {STAGE}")
        time.sleep(90)
        r = sh("pgrep -f '[g]pu_autopilot2' | head -1")
        if not (r.stdout or "").strip():
            log("启动 autopilot2 (从 R8 起)")
            sh("cd /root && setsid nohup python3 -u /root/gpu_autopilot2.py "
               "> /root/autopilot2.log 2>&1 < /dev/null &")
            time.sleep(20)
            r2 = sh("pgrep -f '[g]pu_autopilot2' | head -2")
            log("v2 pids: " + (r2.stdout or "").strip().replace("\n", ","))
        else:
            log("v2 已在运行(时间触发先到), 跳过")
        break
    time.sleep(60)
log("watch exit")

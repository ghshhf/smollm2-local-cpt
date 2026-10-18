#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_ensure.py — 幂等「确保 GPU 在跑」（2026-10-18）

作用: 检查 autopilot 是否存活 -> 活着就回报进度; 没活就 nohup 后台拉起。
      重复执行安全。用于「额度在烧, 别让它空转」的场景。

Jupyter 格（无引号版）:
    !curl -s -o /tmp/e.py http://47.99.131.12:8081/file/gpu_ensure.py\?key=k0u4mail && python3 /tmp/e.py
"""
import os, subprocess, time, glob

W = "/mnt/workspace/node1"
CH = W + "/chain"
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
REP = []


def sh(c, timeout=300):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    REP.append(line)


def upload(name):
    try:
        with open("/tmp/" + name, "rb") as f:
            data = f.read()
        import urllib.request
        req = urllib.request.Request(f"{ECS}/upload?name={name}&key={KEY}", data=data, method="POST",
                                     headers={"Content-Type": "application/octet-stream"})
        urllib.request.urlopen(req, timeout=300).read()
    except Exception as e:
        say(f"[upload] FAIL: {repr(e)[:120]}")


say("=== GPU ENSURE ===")
r = sh("ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E 'g.py|autopilot|train_remote_v5' | grep -v grep | head -6")
alive = [l for l in (r.stdout or "").strip().split("\n") if l.strip()]
say("存活进程: " + (("; ".join(alive))[:500] if alive else "无"))

if not alive:
    say("!! autopilot 不在跑 -> 后台拉起")
    sh(f"curl -s -o /root/gpu_autopilot.py '{ECS}/file/gpu_autopilot.py?key={KEY}'")
    sh("cd /root && setsid nohup python3 -u /root/gpu_autopilot.py > /root/autopilot.log 2>&1 < /dev/null &")
    time.sleep(20)
    r = sh("ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E 'autopilot|train_remote_v5' | grep -v grep | head -4")
    say("拉起后: " + (r.stdout.strip()[:400] or "!! 仍未起来"))
    r = sh("tail -12 /root/autopilot.log 2>/dev/null")
    say("日志: " + (r.stdout.strip()[-800:] or "无"))
else:
    say("autopilot 在跑, 汇报进度")

st = ""
p = CH + "/ckpt_gpu/checkpoint-latest/step.txt"
if os.path.exists(p):
    st = open(p).read().strip()
say("NAS ckpt_gpu 最新 step = " + (st or "尚无"))
r = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
say("GPU: " + (r.stdout.strip() or "?"))
if os.path.exists("/root/autopilot.log"):
    r = sh("tail -6 /root/autopilot.log")
    say("autopilot.log 尾: " + r.stdout.strip()[-500:])

open("/tmp/gpu_ensure_report.txt", "w", encoding="utf-8").write("\n".join(REP))
upload("gpu_ensure_report.txt")
say("DONE")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_mig.py — 迁移前侦察: 完整进程树/有无守护者/当前 chunk 进度/pid 锁状态。
结果写 <NAME>.txt (文件名轮换, 避 ECS 文件锁)。
"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
W = "/mnt/workspace/node1/chain"
NAME = sys.argv[1] if len(sys.argv) > 1 else "mig1.txt"

CMD = (
    "cd " + W + "; "
    "{ echo '#### TIME'; date '+%m-%d %H:%M:%S'; "
    "echo '#### PS_ALL(chain/train/boot/agent/cron/python)'; "
    "ps -eo pid,ppid,etime,cmd | grep -E '[c]hain_v|[t]rain_remote_v4|[b]oot|[n]ode_v|[a]gent|[c]ron|[w]atch|[s]upervis' ; "
    "echo '#### CRONTAB'; crontab -l 2>&1 | head -20; "
    "echo '#### PS_TOTAL'; ps -e | wc -l; "
    "echo '#### W_DIR'; ls -la . | head -30; "
    "echo '#### PID_LOCK'; cat chain.pid 2>/dev/null || echo 'no chain.pid(老链无锁, 正常)'; "
    "echo '#### HEARTBEAT'; cat heartbeat 2>/dev/null || echo 'no heartbeat'; "
    "echo '#### STOP'; if [ -f stop ]; then echo present; else echo absent; fi; "
    "echo '#### DONE'; cat done_steps.txt 2>/dev/null; "
    "echo '#### CHUNK_PROGRESS(最近 step 行)'; grep '^step ' chain.log | tail -3; "
    "echo '#### LOG_TAIL'; tail -8 chain.log; "
    "echo '#### CKPT'; ls -la ckpt/ | head -12; "
    "} 2>&1 | curl -s -X POST '" + ECS + "/upload?name=" + NAME + "&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "mig-" + NAME.replace(".txt", ""),
                   "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
except Exception as e:
    print("POST ERR:", e)

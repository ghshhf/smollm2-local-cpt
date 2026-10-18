#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stop_cpu_train.py — 停掉 node1(CPU) 的 8 tok/s 训练，保留 node_v5 中继。

理由: GPU A10 实测 9566 tok/s (1196倍), CPU 链产出无价值且会写 chain/ckpt 与
      done_steps, 与 GPU 的 chain/ckpt_gpu 状态打架。
保留: node_v5.py (领活/心跳/回传中继), 供后续轻量任务(语料预处理/诊断)使用。
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
NODE = "/mnt/workspace/node1"

CMD = (
    "{ echo '=== BEFORE ==='; ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E 'train_remote|chain_v2|node_v5' | grep -v grep | head -8; "
    "echo '=== KILL ==='; pkill -f train_remote_v4.py; pkill -f chain_v2.sh; sleep 4; "
    "echo '=== AFTER ==='; ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E 'train_remote|chain_v2|node_v5|python3' | grep -v grep | head -8; "
    "echo '=== STATE ==='; cat %s/chain/done_steps.txt 2>/dev/null; cat %s/chain/heartbeat 2>/dev/null; "
    "echo '=== GPU_CKPT ==='; ls -lt %s/chain/ckpt_gpu/ 2>/dev/null | head -5; "
    "echo '=== CORPUS ==='; du -sh %s/chain/corpus_R7 2>/dev/null; ls %s/chain/corpus_R7 2>/dev/null | head -5; "
    "echo '=== LOAD ==='; uptime; } 2>&1 | "
    "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=cpu_stop.txt&key=%s' --data-binary @-"
) % (NODE, NODE, NODE, NODE, NODE, KEY)

body = json.dumps({"cmd": CMD, "id": "stop-cpu-%d" % int(time.time()), "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
print("POST ->", urllib.request.urlopen(req, timeout=20).read().decode()[:200])

txt = ""
for _ in range(20):
    time.sleep(4)
    try:
        txt = urllib.request.urlopen(
            "%s/file/cpu_stop.txt?key=%s" % (ECS, KEY), timeout=30
        ).read().decode("utf-8", "replace")
    except Exception:
        txt = ""
    if "=== LOAD ===" in txt:
        break
print(txt if txt.strip() else "(无回传)")

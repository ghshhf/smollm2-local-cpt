#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_env.py — 探 node1 环境：训练链状态 / 核数负载 / 135M 权重 / eval 脚本 / tokenizer / torch。
结果由 node1 自己 curl 回传 ECS /upload，本机再 curl ECS /file/<name> 取回。
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
ts = int(time.time())
name = f"probe_env_{ts}.txt"

CMD = (
    "W=$(ls -d /mnt/workspace/node1/chain 2>/dev/null | head -1); "
    "{ "
    "echo '=== date ==='; date; "
    "echo '=== chain ==='; P=$(cat \"$W/chain.pid\" 2>/dev/null); echo \"pid=$P\"; "
    "{ [ -n \"$P\" ] && kill -0 \"$P\" 2>/dev/null && echo CHAIN_ALIVE; } || echo CHAIN_DEAD; "
    "echo '=== stage ==='; grep -oE 'stage=[A-Z0-9]+' \"$W/heartbeat\" 2>/dev/null; "
    "echo '=== nproc ==='; nproc; "
    "echo '=== loadavg ==='; cat /proc/loadavg; "
    "echo '=== 135m ckpt ==='; find /mnt/workspace -maxdepth 6 -path '*out_smollm2_135m_pt*' -name 'pytorch_model.bin' 2>/dev/null | head; "
    "echo '=== eval v1_02 ==='; find /mnt/workspace -maxdepth 6 -name 'v1_02_alpha_ladder.py' 2>/dev/null | head; "
    "echo '=== eval e1 ==='; find /mnt/workspace -maxdepth 6 -name 'e1_subliminal_proxy.py' 2>/dev/null | head; "
    "echo '=== SmolLM2-200M dir ==='; find /mnt/workspace -maxdepth 6 -type d -name 'SmolLM2-200M' 2>/dev/null | head; "
    "echo '=== torch ==='; python3 -c 'import torch;print(torch.__version__)' 2>&1 | head -1; "
    "} 2>&1 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=" + name + "&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": f"probe-env-{ts}", "ts": ts}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
    print("fetch URL:", f"{ECS}/file/{name}")
except Exception as e:
    print("POST ERR:", e)

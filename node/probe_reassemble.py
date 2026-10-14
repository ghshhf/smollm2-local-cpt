#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_reassemble.py — 查 node1 部署命令的中间状态：重组进度 / 脚本下载 / 摆路径结果。"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
ts = int(time.time())
out_name = f"probe_reasm_{ts}.txt"

CMD = (
    "G=/mnt/workspace/node1/gh_repo; "
    "{ "
    "echo '=== 0.1B.zip (重组产物, 正在写?) ==='; ls -la \"$G/backups/0.1B/0.1B.zip\" 2>&1; "
    "echo '=== extracted ckpt ==='; ls -la \"$G/backups/0.1B/0.1B_extracted/model/pytorch_model.bin\" 2>&1; "
    "echo '=== staged 135M ckpt ==='; ls -la \"$G/out_smollm2_135m_pt/checkpoint-latest/pytorch_model.bin\" 2>&1; "
    "echo '=== staged tokenizer ==='; ls \"$G/models/SmolLM2-200M/\" 2>&1; "
    "echo '=== eval scripts ==='; ls -la \"$G/eval/\"*.py 2>&1; "
    "echo '=== python procs ==='; ps -eo pid,etimes,args 2>/dev/null | grep -E 'reassemble|e1_subliminal|v1_02|setup' | grep -v grep | head; "
    "} 2>&1 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=" + out_name + "&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": f"probe-reasm-{ts}", "ts": ts}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
    print("fetch:", f"{ECS}/file/{out_name}")
except Exception as e:
    print("POST ERR:", e)

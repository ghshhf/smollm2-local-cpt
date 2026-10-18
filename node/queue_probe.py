#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""queue_probe.py — 查远程档位队列(R7/R8/...)+0.4B 语料是否到位+扩容脚本。"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
CH = "/mnt/workspace/node1/chain"
ND = "/mnt/workspace/node1"

PARTS = [
    "echo HOST=$(hostname); date '+%H:%M:%S'",
    "echo '=== 档位语料目录 ==='; ls -d " + CH + "/corpus_* 2>/dev/null",
    "for d in " + CH + "/corpus_*; do echo -n \"$d: \"; du -sh $d 2>/dev/null | cut -f1; ls $d 2>/dev/null | head -3; done",
    "echo '=== 0.4B 相关 ==='; ls -d " + ND + "/*04b* " + ND + "/*0_4* " + CH + "/*04b* 2>/dev/null | head -10",
    "echo '=== node1 顶层 ==='; ls -lt " + ND + " 2>/dev/null | head -16",
    "echo '=== 进度状态 ==='; cat " + CH + "/done_steps.txt 2>/dev/null; cat " + CH + "/heartbeat 2>/dev/null",
    "echo '=== chain.log 档位推进 ==='; grep -E 'R[0-9]+ (complete|done|start)|0\\.[34]B|扩容' " + CH + "/chain.log 2>/dev/null | tail -12",
    "echo '=== 扩容脚本 ==='; ls -la " + CH + "/expand*.py 2>/dev/null | head -5",
    "echo '=== chain_v2.sh 档位逻辑 ==='; grep -nE 'corpus_R|R[0-9]+|0\\.[34]B|expand' " + CH + "/chain_v2.sh 2>/dev/null | head -18",
    "echo '=== GPU ckpt ==='; ls -lt " + CH + "/ckpt_gpu/ 2>/dev/null | head -4; cat " + CH + "/ckpt_gpu/checkpoint-latest/step.txt 2>/dev/null",
    "echo '=== GPU ==='; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
]
CMD = ("{ " + "; ".join(PARTS) + "; } 2>&1 | curl -s -m90 -X POST '"
       + ECS + "/upload?name=queue_probe.txt&key=" + KEY + "' --data-binary @-")

body = json.dumps({"cmd": CMD, "id": "qprobe-%d" % int(time.time()), "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
urllib.request.urlopen(req, timeout=20).read()

txt = ""
for _ in range(24):
    time.sleep(5)
    try:
        txt = urllib.request.urlopen(
            "%s/file/queue_probe.txt?key=%s" % (ECS, KEY), timeout=40
        ).read().decode("utf-8", "replace")
    except Exception:
        txt = ""
    if "=== GPU ===" in txt:
        break
print(txt if txt.strip() else "(无回传)")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_clean_apply.py — 派 node1 执行清洗并原地替换(只剔模板残留+去重)。

关键: 训练器按 corpus_dir/*.jsonl 全扫, 原文件 **必须移出 corpus 目录** 再备份,
      否则清洗后的新文件和原文件会被一起读进去, 污染又回来了。

结果写 clean_apply<N>.txt (文件名轮换)。
"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
W = "/mnt/workspace/node1/chain"
NAME = sys.argv[1] if len(sys.argv) > 1 else "clean_apply1.txt"

TARGETS = [
    ("corpus_r5", "zh_text_r5.jsonl"),
    ("corpus_r6", "zh_text_r6.jsonl"),
    ("corpus_reserve", "zh_r_novels.jsonl"),
    ("corpus_reserve", "zh_r_weixin.jsonl"),
    ("corpus_reserve", "zh_r_ancient.jsonl"),
    ("corpus_reserve", "zh_r_knowledge.jsonl"),
]

steps = ["mkdir -p " + W + "/backup_corpus"]
for d, fn in TARGETS:
    src = f"{W}/{d}/{fn}"
    tmp = f"{W}/{d}/{fn}.clean"
    bak = f"{W}/backup_corpus/{d}__{fn}"
    steps.append(f"echo 'CLEAN {d}/{fn}'")
    steps.append(f"python3 {W}/clean_corpus.py {src} {tmp}")
    steps.append(f"mv -f {src} {bak}")
    steps.append(f"mv -f {tmp} {src}")
steps.append("echo 'VERIFY 目录内不得残留 .orig/.clean'")
steps.append(f"ls {W}/corpus_r5/ {W}/corpus_r6/ | grep -E 'orig|clean' || echo 'clean-ok'")
steps.append(f"ls {W}/corpus_reserve/ | grep -E 'orig|clean' || echo 'reserve-ok'")

CMD = ("cd " + W + "; { " + "; ".join(steps) + "; "
       "} 2>&1 | curl -s -X POST '" + ECS + "/upload?name=" + NAME + "&key=k0u4mail' --data-binary @-")

body = json.dumps({"cmd": CMD, "id": "apply-" + NAME.replace(".txt", ""),
                   "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
except Exception as e:
    print("POST ERR:", e)

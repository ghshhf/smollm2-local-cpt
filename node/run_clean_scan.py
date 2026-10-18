#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_clean_scan.py — 派 node1: 拉清洗脚本 -> 报停机状态 -> 扫描各语料污染量(只统计不改)。
结果写 clean_scan<N>.txt (文件名轮换, 避 ECS 文件锁)。
"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
W = "/mnt/workspace/node1/chain"
NAME = sys.argv[1] if len(sys.argv) > 1 else "clean_scan1.txt"
TARGETS = sys.argv[2:] or ["corpus_r5/zh_text_r5.jsonl"]

scans = "; ".join(
    f"echo 'SCAN {t}'; python3 {W}/clean_corpus.py --scan {W}/{t}"
    for t in TARGETS)

CMD = (
    "curl -s -m120 -o " + W + "/clean_corpus.py " + ECS + "/file/clean_corpus.py; "
    "cd " + W + "; "
    "{ echo TIME; date '+%m-%d %H:%M:%S'; "
    "echo DONE_STEPS; cat done_steps.txt 2>/dev/null; "
    "echo STOP_FLAG; if [ -f stop ]; then echo present; else echo absent; fi; "
    "echo ALIVE_PROC; ps aux|grep -E '[c]hain_v1|[t]rain_remote_v4' | wc -l; "
    "echo R5_CACHE; ls -la /root/c_cache_R5.npy 2>/dev/null || echo 'no r5 cache'; "
    + scans + "; "
    "} | curl -s -X POST '" + ECS + "/upload?name=" + NAME + "&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "scan-" + NAME.replace(".txt", ""),
                   "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
    print("targets:", TARGETS)
except Exception as e:
    print("POST ERR:", e)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_r7_threads.py — node1 上测 8线程 vs 64线程 的 GEMM 吞吐，决定是否值得重启训练。"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"

BENCH = r'''
import torch, time, json
res = {}
for n in (8, 16, 64):
    torch.set_num_threads(n)
    # 模拟训练真实 GEMM: hidden 2304 (0.3B inter=4608/2), seq 512
    a = torch.randn(512, 2304); w = torch.randn(2304, 2304)
    a@w  # warmup
    t = time.time(); K = 5
    for _ in range(K): a@w
    dt = (time.time()-t)/K
    flops = 2*512*2304*2304/dt/1e9
    res[n] = {"sec": round(dt,3), "gflops": round(flops,1)}
    print(n, "threads:", round(dt,3), "s  =", round(flops,1), "GFLOPS", flush=True)
# 大矩阵(更接近 MLP 实况)
for n in (8, 64):
    torch.set_num_threads(n)
    a = torch.randn(4096, 4096); b = torch.randn(4096, 4096)
    a@b
    t = time.time(); K = 3
    for _ in range(K): a@b
    dt = (time.time()-t)/K
    flops = 2*4096**3/dt/1e9
    res[f"big{n}"] = {"sec": round(dt,3), "gflops": round(flops,1)}
    print("4096^3", n, "threads:", round(dt,3), "s =", round(flops,1), "GFLOPS", flush=True)
print("CONCLUSION", json.dumps(res), flush=True)
'''

import base64
B64 = base64.b64encode(BENCH.encode()).decode()

CMD = (
    "echo '%s' | base64 -d > /tmp/bench_threads.py && "
    "timeout 240 python3 /tmp/bench_threads.py 2>&1 | "
    "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=bench_threads_result.txt&key=%s' --data-binary @-"
) % (B64, KEY)

body = json.dumps({"cmd": CMD, "id": "bench-threads-%d" % int(time.time()), "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
print("POST ->", urllib.request.urlopen(req, timeout=20).read().decode()[:200])

txt = ""
for _ in range(24):
    time.sleep(5)
    try:
        txt = urllib.request.urlopen(
            "%s/file/bench_threads_result.txt?key=%s" % (ECS, KEY), timeout=40
        ).read().decode("utf-8", "replace")
    except Exception:
        txt = ""
    if "CONCLUSION" in txt:
        break
print(txt if txt.strip() else "(无回传)")

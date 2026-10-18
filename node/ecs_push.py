#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ecs_push.py — 把本地文件推到 ECS 信箱，供 node1 / GPU 实例 curl 拉取。
用法: python ecs_push.py <本地文件> [远端名]
     python ecs_push.py train_remote_v5_gpu.py gpu_quickstart.py
"""
import os, sys, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"


def push(path, name=None):
    name = name or os.path.basename(path)
    with open(path, "rb") as f:
        data = f.read()
    req = urllib.request.Request(f"{ECS}/upload?name={name}&key={KEY}", data=data, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    out = urllib.request.urlopen(req, timeout=300).read().decode("utf-8", "ignore")
    print(f"[push] {os.path.basename(path)} -> {name} ({len(data)/1024:.1f} KB) : {out[:120]}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        push(p)

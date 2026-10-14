#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ecs_upload.py — 把本地文件 POST 到 ECS /upload（绕开大文件锁：文件名自带时间戳轮换）。
用法: python ecs_upload.py <local_path> [ecs_name]
返回最终 ecs_name（供下游 node1 curl 拉取）。
"""
import sys, time, urllib.request

ECS = "http://47.99.131.12:8081"

def upload(local, name=None):
    if name is None:
        ts = int(time.time())
        name = f"n1_{ts}_{local.split('/')[-1].split('\\\\')[-1]}"
    with open(local, "rb") as f:
        data = f.read()
    req = urllib.request.Request(f"{ECS}/upload?name={name}&key=k0u4mail", data=data, method="POST")
    try:
        r = urllib.request.urlopen(req, timeout=120)
        print(f"OK name={name} status={r.status} bytes={len(data)} resp={r.read().decode()[:120]}")
        return name
    except Exception as e:
        print("ERR", e)
        return None

if __name__ == "__main__":
    local = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else None
    n = upload(local, name)
    if n:
        print("ECS_NAME=" + n)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n1_push.py — 把本地文件推到 ECS /file/<name> (带重试, 绕开信箱偶发 wedge)。
用法: python3 n1_push.py <local_path> <remote_name>
"""
import sys, time, urllib.request

ECS = "http://47.99.131.12:8081"

def push(local, name, retries=6):
    data = open(local, "rb").read()
    # 行尾归一化: Windows 下编辑出的脚常是 CRLF, 而 node1 的 /bin/sh 是 dash,
    # 会把行尾的 \r 当普通字符 -> "Syntax error: newline unexpected"。
    # 文本文件(无 NUL 字节)统一按 LF 上传。
    if b"\r\n" in data and b"\x00" not in data:
        data = data.replace(b"\r\n", b"\n")
        print(f"[push] {name}: CRLF -> LF 已归一化")
    for i in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                f"{ECS}/upload?name={urllib.parse.quote(name)}&key=k0u4mail",
                data=data, method="POST",
                headers={"Content-Type": "application/octet-stream"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read(200).decode("utf-8", "ignore")
                print(f"[push] {name} -> {r.status} {body}")
            return True
        except Exception as e:
            print(f"[push] FAIL {name} (try {i}/{retries}): {e}")
            if i < retries:
                time.sleep(8 * i)
    return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: n1_push.py <local> <remote_name>"); sys.exit(2)
    ok = push(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)

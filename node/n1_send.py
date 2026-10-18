#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n1_send.py — 把一段 shell 命令派给 node1 (POST /task), 规避 shell 引号地狱。
用法: python n1_send.py <cmd_file>      或   python n1_send.py -  (从 stdin 读)
"""
import json, sys, urllib.request

ECS = "http://47.99.131.12:8081"

def send(cmd, timeout=20):
    req = urllib.request.Request(f"{ECS}/task",
        data=json.dumps({"cmd": cmd.strip()}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "-"
    if src == "-":
        cmd = sys.stdin.read()
    else:
        cmd = open(src, encoding="utf-8").read()
    print(send(cmd))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n1_exec.py — 经 ECS 信箱把一段 shell 交给 node1 执行并取回输出

为什么需要它(2026-10-19):
  node1 只吃单行命令; 在 bash -> python -c -> 远程 shell 三层里嵌引号必然崩。
  本工具把命令写进本地文件, 由 Python 读入后原样 POST, 彻底绕开转义。

用法:
  python tools/n1_exec.py -f cmd.sh                 # 从文件读命令
  python tools/n1_exec.py -f cmd.sh --timeout 60
  python tools/n1_exec.py -c "echo hi"              # 直接给命令(短命令才用)

约定:
  命令里自行把输出喂给 curl 上传; 本工具会自动补一段兜底:
  若用户命令没有 '| curl', 自动包成 `{ ...; } 2>&1 | curl -s --data-binary @- <upload>`
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"


def post(cmd):
    body = json.dumps({"cmd": cmd, "key": KEY}).encode("utf-8")
    req = urllib.request.Request(ECS + "/task", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status, r.read().decode("utf-8", "ignore")


def fetch(name):
    url = "%s/file/%s?key=%s" % (ECS, urllib.parse.quote(name), KEY)
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            if r.status != 200:
                return None
            return r.read().decode("utf-8", "ignore")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file")
    ap.add_argument("-c", "--cmd")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--tag", default=None, help="回传文件名, 默认按时间生成")
    a = ap.parse_args()

    if a.file:
        cmd = open(a.file, encoding="utf-8").read().strip()
    elif a.cmd:
        cmd = a.cmd.strip()
    else:
        sys.exit("[err] 需要 -f 或 -c")

    tag = a.tag or ("ex_%s.txt" % time.strftime("%H%M%S"))

    if "| curl" not in cmd and "--data-binary" not in cmd:
        cmd = "{ %s\n} 2>&1 | curl -s -m 30 --data-binary @- '%s/upload?name=%s&key=%s'" % (
            cmd, ECS, tag, KEY)

    st, txt = post(cmd)
    print("[post] %s %s  tag=%s" % (st, txt[:80], tag), flush=True)

    t0 = time.time()
    while time.time() - t0 < a.timeout:
        time.sleep(2.5)
        out = fetch(tag)
        if out is not None and len(out) > 0:
            print(out)
            return
    print("[err] %.0fs 内没有回传 (tag=%s)" % (a.timeout, tag))


if __name__ == "__main__":
    main()

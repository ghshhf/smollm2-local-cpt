#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_live_push.py — node1 常驻：每 60s 把训练实时状态覆盖上传 ECS 固定文件。

上传目标：POST http://47.99.131.12:8081/upload?name=train_live.txt&key=k0u4mail
外部读取：GET  http://47.99.131.12:8081/file/train_live.txt?key=k0u4mail
（ECS 同名文件为覆盖写，已实测；node_v5 老轮询器或 watchd 均可常驻运行本脚本）
"""
import os
import re
import subprocess
import time
import urllib.request

NODE = "/mnt/workspace/node1"
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
NAME = "train_live.txt"
INTERVAL = 60


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def collect():
    done = sh("cat %s/chain/done_steps.txt 2>/dev/null" % NODE)
    tail = sh("tail -3 %s/chain/chain.log 2>/dev/null" % NODE)
    procs = sh("pgrep -c -f train_remote_v4 2>/dev/null || echo 0")
    ckpt = sh("ls -t %s/chain/ckpt/ 2>/dev/null | head -3 | tr '\\n' ' '" % NODE)
    hb = sh("cat %s/chain/heartbeat 2>/dev/null" % NODE)
    lines = [
        "# node1 train live status (auto, 60s)",
        "TS=%s" % time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "STAGE=R6",
        "DONE_STEPS=%s" % (done or "?"),
        "TARGET=6800",
        "TRAIN_PROCS=%s" % (procs or "0"),
        "HEARTBEAT=%s" % (hb or "-"),
        "CKPT_LATEST=%s" % (ckpt or "-"),
        "CHAIN_LOG_TAIL:",
    ]
    lines += [l for l in tail.splitlines() if l.strip()]
    txt = "\n".join(lines) + "\n"
    # 从末行解析关键指标，便于 AI 直接读
    m = re.findall(r"step (\d+)/(\d+) loss=([\d.]+) lr=([\d.e-]+) t=(\d+)s tok/s=(\d+)", tail)
    if m:
        s, tot, loss, lr, t, toks = m[-1]
        sps = round(int(t) / max(1, int(s)), 2)
        eta = int(max(0, int(tot) - int(s)) * sps)
        lines2 = [
            "",
            "PARSED:",
            "CHUNK=%s/%s" % (s, tot),
            "LOSS=%s" % loss,
            "LR=%s" % lr,
            "TOK_S=%s" % toks,
            "SEC_PER_STEP=%s" % sps,
            "ETA_SECONDS=%d" % eta,
            "ETA_HUMAN=%dh%02dm" % divmod(eta // 60, 60),
        ]
        txt += "\n".join(lines2) + "\n"
    return txt


def upload(txt):
    data = txt.encode("utf-8")
    url = "%s/upload?name=%s&key=%s" % (ECS, NAME, KEY)
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")[:80]


def main():
    print("[live] push start -> %s/%s every %ds" % (ECS, NAME, INTERVAL), flush=True)
    while True:
        try:
            txt = collect()
            print("[live] %s -> %s" % (time.strftime("%H:%M:%S"), upload(txt)), flush=True)
        except Exception as e:
            print("[live] err %s" % e, flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()

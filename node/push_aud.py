#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""push_aud.py — 把打包好的音频分片经 ECS 信箱推到 node1 可取的位置。

用法: python push_aud.py [--dir /tmp/audpack] [--prefix aud]
每个分片 POST /upload?name=<prefix>_<part>，node1 侧按同名 GET /file/<name>。
实测 ECS 上传 ~5MB/s。
"""
import argparse
import os
import sys
import time
import urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"


def push_one(path, name):
    with open(path, "rb") as f:
        data = f.read()
    url = "%s/upload?name=%s&key=%s" % (ECS, name, KEY)
    req = urllib.request.Request(url, data=data, method="POST")
    t = time.time()
    urllib.request.urlopen(req, timeout=180).read()
    dt = time.time() - t
    return len(data), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/audpack")
    ap.add_argument("--prefix", default="aud")
    a = ap.parse_args()

    parts = sorted(f for f in os.listdir(a.dir) if f.startswith("p_"))
    if not parts:
        print("[fatal] %s 下没有分片" % a.dir)
        return 2
    total = sum(os.path.getsize(os.path.join(a.dir, p)) for p in parts)
    print("[push] %d 分片 / %.0fMB -> %s" % (len(parts), total / 1e6, ECS), flush=True)

    t0 = time.time()
    done = 0
    for i, p in enumerate(parts, 1):
        full = os.path.join(a.dir, p)
        name = "%s_%s" % (a.prefix, p)
        try:
            n, dt = push_one(full, name)
            done += n
            el = time.time() - t0
            print("  [%d/%d] %s %.1fMB %.1fs (%.1fMB/s)  累计 %.0f/%.0fMB"
                  % (i, len(parts), p, n / 1e6, dt, n / 1e6 / max(dt, 0.1),
                     done / 1e6, total / 1e6), flush=True)
        except Exception as e:
            print("  [FAIL] %s %r" % (p, e), flush=True)
            return 1
    print("[push] 完成 %.0fs 共 %.0fMB" % (time.time() - t0, done / 1e6), flush=True)
    print("[push] 分片命名: %s_%s ... %s_%s" % (a.prefix, parts[0], a.prefix, parts[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/fetch_04b_data.py — 0.4B 代码 / 音频原始素材下载（魔搭 CDN，断点续传）

node1 实测 79.4MB/s。落盘 /mnt/workspace/node1/data/（NAS，实例回收不丢）。

用法:
    python3 tools/fetch_04b_data.py --what code
    python3 tools/fetch_04b_data.py --what audio
    python3 tools/fetch_04b_data.py --what all
"""
import argparse
import os
import sys
import time
import urllib.error
import urllib.request

W = "/mnt/workspace/node1/data"
URL = "https://modelscope.cn/api/v1/datasets/%s/repo?Revision=master&FilePath=%s"

# 🔴 分块下载（2026-10-19 实测）：魔搭 CDN 的新连接 68~100MB/s，
# 但同一个长连接读到中途会被限速到 1MB/s。所以每块重开一次连接。
# 32MB 块 = 1000 次连接（FMA 31.9GB），64MB 块 = 500 次，握手摊薄一半。
CHUNK = 64 << 20

CODE = [("MoreThought/SmolPyEdu", "data/train-%05d-of-00010.parquet" % i) for i in range(10)]
AUDIO = [("OmniData/FMA", "raw/FMA.tar.gz")]


def size_of(p):
    return os.path.getsize(p) if os.path.exists(p) else 0


def _total_of(url, out):
    """拿总长度：优先 sidecar，否则用 Range:0-0 探测（不下载正文）。"""
    side = out + ".size"
    if os.path.exists(side):
        try:
            return int(open(side).read().strip())
        except Exception:
            pass
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        r = urllib.request.urlopen(req, timeout=60)
        cr = r.headers.get("Content-Range") or ""
        r.close()
        if "/" in cr:
            n = int(cr.rsplit("/", 1)[1])
            open(side, "w").write(str(n))
            return n
    except Exception as e:
        print("  [size?] %s" % e, flush=True)
    return None


def fetch(repo, path, outdir):
    name = os.path.basename(path)
    out = os.path.join(outdir, name)
    os.makedirs(outdir, exist_ok=True)
    url = URL % (repo, path)
    t0 = time.time()
    total_expect = _total_of(url, out)
    if total_expect and size_of(out) >= total_expect:
        print("[OK] %s %.2fGB (已完整)" % (out, size_of(out) / 1e9), flush=True)
        return True
    print("  %s 目标 %.2fGB，已有 %.2fGB" % (
        name, (total_expect or 0) / 1e9, size_of(out) / 1e9), flush=True)

    for attempt in range(1, 13):
        try:
            while True:
                have = size_of(out)
                if total_expect and have >= total_expect:
                    break
                req = urllib.request.Request(url)
                req.add_header("Range", "bytes=%d-%d" % (have, have + CHUNK - 1))
                try:
                    r = urllib.request.urlopen(req, timeout=90)
                except urllib.error.HTTPError as e:
                    if e.code == 416:
                        total_expect = have
                        break
                    raise
                if r.status not in (200, 206):
                    break
                crange = r.headers.get("Content-Range") or ""
                if total_expect is None and "/" in crange:
                    try:
                        total_expect = int(crange.rsplit("/", 1)[1])
                        open(out + ".size", "w").write(str(total_expect))
                    except ValueError:
                        pass
                got = 0
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(out, mode) as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
                r.close()
                if got == 0:
                    break
                cur = have + got
                dt = max(time.time() - t0, 1e-6)
                pct = 100.0 * cur / total_expect if total_expect else 0
                print("  %s %.1f%% %.1fMB/s %.2fGB" % (
                    name, pct, cur / dt / 1e6, cur / 1e9), flush=True)
            if total_expect and size_of(out) < total_expect:
                raise IOError("incomplete %d/%d" % (size_of(out), total_expect))
            print("[OK] %s %.2fGB %.0fs" % (out, size_of(out) / 1e9, time.time() - t0), flush=True)
            return True
        except Exception as e:
            print("[RETRY %d] %s %s" % (attempt, name, e), flush=True)
            time.sleep(min(20, 2 * attempt))
    print("[FAIL] %s" % name, flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all", choices=["code", "audio", "all"])
    ap.add_argument("--repo", help="指定单源: 命名空间/数据集")
    ap.add_argument("--path", help="指定单源: 仓库内文件路径")
    ap.add_argument("--out", default="code_raw", help="单源落盘目录(相对 data/)")
    a = ap.parse_args()
    if a.repo and a.path:
        return 0 if fetch(a.repo, a.path, os.path.join(W, a.out)) else 1
    ok = True
    if a.what in ("code", "all"):
        print("== 代码: SmolPyEdu 10 片（纯 Python，GitHub 经 EDU 过滤）==", flush=True)
        for repo, p in CODE:
            ok &= fetch(repo, p, os.path.join(W, "code_raw", "smolpyedu"))
    if a.what in ("audio", "all"):
        print("== 音频: FMA 全量（106,574 首 / 343 天）==", flush=True)
        for repo, p in AUDIO:
            ok &= fetch(repo, p, os.path.join(W, "audio_raw", "fma"))
    print("== DONE ok=%s ==" % ok, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

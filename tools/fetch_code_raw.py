#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/fetch_code_raw.py — 拉取代码原始素材（可断点续传）。

只做一件事：把候选源下载到 data/code_raw/，不做任何加工。
筛选在 tools/filter_code.py 里，两步分开，出错能单独重跑。

用法：
    python tools/fetch_code_raw.py --list          # 只看要下什么、多大
    python tools/fetch_code_raw.py                 # 全下
    python tools/fetch_code_raw.py --only magicoder
    python tools/fetch_code_raw.py --raw-cap-gb 4  # codeparrot 原始码上限
"""
import argparse
import json
import os
import sys
import time
import urllib.request

PROXY = os.environ.get("POLARIS_PROXY", "")          # 空 = 直连（node1 上直连镜像）
BASE = os.environ.get("POLARIS_HF", "https://hf-mirror.com")   # 国内镜像优先
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "code_raw")

# (简名, repo, 文件, 用途)
INSTRUCT = [
    ("magicoder_evol", "ise-uiuc/Magicoder-Evol-Instruct-110K",
     "data-evol_instruct-decontaminated.jsonl", "指令→代码，110k"),
    ("magicoder_oss", "ise-uiuc/magicoder-oss-instruct-75k",
     "data-oss_instruct-decontaminated.jsonl", "指令→代码，75k"),
    ("codefeedback", "m-a-p/CodeFeedback-Filtered-Instruction",
     "CodeFeedback-Filtered-Instruction.jsonl", "多轮修正，156k"),
    ("codefuse_evol", "codefuse-ai/Evol-instruction-66k",
     "Evol-Instruction-66k.json", "中文指令，66k"),
    ("codefuse_exer", "codefuse-ai/CodeExercise-Python-27k",
     "CodeExercise-Python-27k.json", "中文题面+用例，27k"),
    ("apps_train", "codeparrot/apps", "train.jsonl", "题面+测试用例"),
]


def _op():
    if PROXY:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
    return urllib.request.build_opener()


def _url(repo, fn):
    return "%s/datasets/%s/resolve/main/%s" % (BASE, repo, fn)


def _api(path):
    return "%s/api/datasets/%s" % (BASE, path)


def _get(url, timeout=120):
    return _op().open(url, timeout=timeout).read()


def _download(url, path, expect=None):
    """断点续传；返回最终字节数。"""
    tmp = path + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if expect and have >= expect:
        os.replace(tmp, path)
        return have
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", "bytes=%d-" % have)
    t0 = time.time()
    with _op().open(req, timeout=300) as r, open(tmp, "ab") as f:
        total = int(r.headers.get("Content-Length") or 0) + have
        done = have
        last = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if time.time() - last > 5:
                last = time.time()
                sp = (done - have) / max(time.time() - t0, 1) / 1e6
                print("    %5.1f%%  %.1f MB  %.1f MB/s"
                      % (100.0 * done / total if total else 0, done / 1e6, sp), flush=True)
    os.replace(tmp, path)
    return os.path.getsize(path)


def list_targets(raw_cap_gb):
    print("%-16s %-46s %8s  %s" % ("简名", "来源", "大小", "用途"))
    tot = 0
    for name, repo, fn, why in INSTRUCT:
        try:
            n = int(_op().open(_url(repo, fn), timeout=60)
                    .headers.get("Content-Length") or 0)
        except Exception:
            n = 0
        tot += n
        print("%-16s %-46s %7.0fMB  %s" % (name, repo.split("/")[-1], n / 1e6, why))
    print("%-16s %-46s %7.0fMB  %s"
          % ("codeparrot_raw", "codeparrot/codeparrot-clean", raw_cap_gb * 1000,
             "Python 原始仓库代码（按上限取前若干片）"))
    print("合计约 %.1f GB" % (tot / 1e9 + raw_cap_gb))


def fetch_codeparrot(cap_gb):
    """按分片顺序取，直到累计达到上限。"""
    tree = json.loads(_get(_api("codeparrot/codeparrot-clean/tree/main"),
                           timeout=60).decode())
    files = sorted([(x["path"], x.get("size") or 0) for x in tree
                    if x["type"] == "file" and x["path"].endswith(".parquet")])
    if not files:
        print("  [skip] 没找到 parquet 分片")
        return 0
    cap = cap_gb * 1e9
    got = 0
    for p, sz in files:
        if got >= cap:
            break
        dst = os.path.join(OUT, "codeparrot_" + os.path.basename(p))
        if os.path.exists(dst):
            got += os.path.getsize(dst)
            print("  [有] %s  %.0fMB" % (os.path.basename(p), os.path.getsize(dst) / 1e6))
            continue
        print("  [下] %s  %.0fMB" % (os.path.basename(p), sz / 1e6), flush=True)
        n = _download(_url("codeparrot/codeparrot-clean", p), dst, sz)
        got += n
    print("  codeparrot 累计 %.2f GB / %d 片" % (got / 1e9, len(files)))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--raw-cap-gb", type=float, default=4.0)
    a = ap.parse_args()

    if a.list:
        list_targets(a.raw_cap_gb)
        return

    os.makedirs(OUT, exist_ok=True)
    only = set(x.strip() for x in a.only.split(",") if x.strip())

    for name, repo, fn, why in INSTRUCT:
        if only and name not in only:
            continue
        dst = os.path.join(OUT, name + os.path.splitext(fn)[1])
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print("[有] %-16s %.0f MB" % (name, os.path.getsize(dst) / 1e6))
            continue
        print("[下] %-16s %s" % (name, why), flush=True)
        try:
            n = _download(_url(repo, fn), dst)
            print("      -> %.0f MB" % (n / 1e6))
        except Exception as e:
            print("      [失败] %s" % str(e)[:100])

    if not only or "codeparrot" in only:
        print("[下] codeparrot 原始 Python 代码")
        try:
            fetch_codeparrot(a.raw_cap_gb)
        except Exception as e:
            print("  [失败] %s" % str(e)[:100])

    print("\n=== data/code_raw ===")
    tot = 0
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        if os.path.isfile(p):
            tot += os.path.getsize(p)
            print("  %-44s %8.0f MB" % (f, os.path.getsize(p) / 1e6))
    print("合计 %.2f GB" % (tot / 1e9))


if __name__ == "__main__":
    main()

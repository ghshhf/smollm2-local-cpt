# -*- coding: utf-8 -*-
"""dl_aishell.py — 从 hf-mirror 官方镜像下载 AISHELL-1 子集（音频 align 用）。

策略：先下 transcript 全集 + 前 N 个说话人分包（每个 tar.gz 约 25-44MB）。
多连接并行 + curl 断点续传 + 失败重试。只下训练所需子集，不全量 178h。
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://hf-mirror.com/datasets/AISHELL/AISHELL-1/resolve/main"
REPO = "AISHELL/AISHELL-1"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "audio", "raw")
N_SPEAKERS = 24
WORKERS = 3


def list_remote(subdir):
    import json
    import urllib.request
    url = "https://hf-mirror.com/api/datasets/%s/tree/main/%s" % (REPO, subdir)
    with urllib.request.urlopen(url, timeout=30) as r:
        return [f["path"] for f in json.load(r) if f["type"] == "file"]


def dl_one(rel_path):
    dest = os.path.join(OUT, os.path.basename(rel_path))
    url = "%s/%s" % (BASE, rel_path)
    for attempt in range(6):
        cmd = ["curl", "-sL", "--retry", "3", "--retry-delay", "2", "-C", "-",
               "--max-time", "1800", "-o", dest, url]
        rc = subprocess.call(cmd)
        if rc == 0 and os.path.exists(dest) and os.path.getsize(dest) > 1024:
            return rel_path, os.path.getsize(dest), None
        time.sleep(3 * (attempt + 1))
    return rel_path, 0, "failed after retries (rc=%s)" % rc


def main():
    os.makedirs(OUT, exist_ok=True)
    targets = list_remote("data_aishell/wav")[:N_SPEAKERS]
    targets += list_remote("data_aishell/transcript")
    targets += list_remote("resource_aishell")
    total = len(targets)
    print("[dl] %d files -> %s" % (total, OUT), flush=True)

    done = ok = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(dl_one, p): p for p in targets}
        for fut in as_completed(futs):
            p, size, err = fut.result()
            done += 1
            if err:
                print("[FAIL] %s %s" % (p, err), flush=True)
            else:
                ok += 1
                print("[ok %d/%d] %-45s %6.1f MB  (%.0fs)" % (
                    done, total, os.path.basename(p), size / 1e6, time.time() - t0), flush=True)
    print("[DONE] ok=%d fail=%d total_bytes_dir=%d" % (ok, total - ok, sum(
        os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))), flush=True)


if __name__ == "__main__":
    sys.exit(main())

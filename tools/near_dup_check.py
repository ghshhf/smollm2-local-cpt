#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""near_dup_check.py — 正确的近重复检测（修 ufw_audit.py 里 16bit 桶饱和假阳性）。

做法：simhash 64bit + banding 2×32bit（假阳率 ~2e-9），候选对再用汉明距离裁决。
用法：python tools/near_dup_check.py --file corpus_04b/ufw_zh.jsonl --n 20000
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys

WS = re.compile(r"\s+")


def simhash(s):
    v = [0] * 64
    step = 3
    for i in range(0, max(1, len(s) - 4), step):
        hv = int(hashlib.md5(s[i:i + 5].encode("utf-8", "ignore")).hexdigest()[:16], 16)
        for b in range(64):
            v[b] += 1 if (hv >> b) & 1 else -1
    return sum(1 << b for b in range(64) if v[b] > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    sz = os.path.getsize(a.file)
    random.seed(7)
    texts = []
    seen = set()
    with open(a.file, "rb") as f:
        while len(texts) < a.n:
            f.seek(random.randrange(0, max(1, sz - 8192)))
            f.readline()
            line = f.readline()
            if not line:
                continue
            try:
                o = json.loads(line.decode("utf-8", "replace"))
            except Exception:
                continue
            t = WS.sub("", (o.get("text") or o.get("content") or ""))
            if len(t) < 200:
                continue
            h = hashlib.md5(t[:400].encode("utf-8", "ignore")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            texts.append(t[:1200])

    hs = [simhash(t) for t in texts]
    # banding: 2 × 32bit
    b0, b1 = {}, {}
    hits = {3: 0, 6: 0, 10: 0}
    cand = 0
    for i, h in enumerate(hs):
        k0, k1 = h & 0xFFFFFFFF, (h >> 32) & 0xFFFFFFFF
        c = None
        if k0 in b0:
            c = b0[k0]
        elif k1 in b1:
            c = b1[k1]
        if c is not None:
            cand += 1
            d = bin(h ^ hs[c]).count("1")
            for th in (3, 6, 10):
                if d <= th:
                    hits[th] += 1
        b0.setdefault(k0, i)
        b1.setdefault(k1, i)

    n = len(hs)
    print("样本 %d 篇（去重后）" % n)
    print("  LSH 候选对: %d (%.2f%%)" % (cand, cand / n * 100))
    for th in (3, 6, 10):
        print("  汉明距离 <=%-2d 判近重复: %d 篇 (%.2f%%)" % (th, hits[th], hits[th] / n * 100))


if __name__ == "__main__":
    main()

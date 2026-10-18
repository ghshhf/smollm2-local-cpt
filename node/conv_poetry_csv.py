#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""conv_poetry_csv.py — node1 上把 chinese-poetry train.csv（一行一首）转 jsonl。"""
import json, os, time
SRC = "/mnt/workspace/node1/corpus_04b/poetry/train.csv"
OUT = "/mnt/workspace/node1/corpus_04b/poetry_zh.jsonl"
t0 = time.time()
n = tot = 0
with open(SRC, encoding="utf-8-sig") as f:
    f.readline()  # header: text1
    with open(OUT, "w", encoding="utf-8") as out:
        for line in f:
            t = line.strip()
            if len(t) < 10:
                continue
            out.write(json.dumps({"text": t, "source": "poetry"},
                                 ensure_ascii=False) + "\n")
            n += 1
            tot += len(t)
print("[poetry] DONE rows=%d chars=%.1fM -> %s (%.1fMB)"
      % (n, tot / 1e6, OUT, os.path.getsize(OUT) / 1e6), flush=True)

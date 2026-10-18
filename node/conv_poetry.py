#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""conv_poetry.py — node1 上把 chinese-poetry-collection.json 转 jsonl。"""
import json, os, time

SRC = "/mnt/workspace/node1/corpus_04b/poetry/chinese-poetry-collection.json"
OUT = "/mnt/workspace/node1/corpus_04b/poetry_zh.jsonl"
t0 = time.time()
raw = json.load(open(SRC, encoding="utf-8"))

def find_list(o):
    if isinstance(o, list):
        return o
    if isinstance(o, dict):
        for v in o.values():
            r = find_list(v)
            if r is not None:
                return r
    return None

d = find_list(raw)
print("[poetry] loaded %d items (%.1fs)" % (len(d), time.time() - t0), flush=True)
n = 0
tot = 0
with open(OUT, "w", encoding="utf-8") as f:
    for r in d:
        t = (r.get("text1") or "").strip()
        if len(t) < 10:
            continue
        f.write(json.dumps({"text": t, "source": "poetry"}, ensure_ascii=False) + "\n")
        n += 1
        tot += len(t)
print("[poetry] DONE rows=%d chars=%.1fM -> %s (%.1fMB)"
      % (n, tot / 1e6, OUT, os.path.getsize(OUT) / 1e6), flush=True)

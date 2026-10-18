#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_div.py — 诊断精炼产物的来源多样性(同一本书是否霸榜)"""
import json, io, re, collections, os, sys

D = "/mnt/workspace/node1/chain/corpus_reserve_refined"


def book_of(t):
    first = t.split("\n")[0].strip()
    first = re.split(r"第[0-9一二三四五六七八九十百千零〇]+[章節节回]|第[0-9]+[章节]|chapter",
                     first, flags=re.I)[0]
    first = re.sub(r"[（(].*?[)）]", "", first).strip()
    return (first[:24] or "?")


for key in ["knowledge", "ancient", "novels", "weixin"]:
    p = os.path.join(D, "zh_r_%s.jsonl" % key)
    if not os.path.exists(p):
        continue
    c, cs = collections.Counter(), collections.Counter()
    n = 0
    with io.open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("text", "")
            n += 1
            c[book_of(t)] += 1
            cs[str(d.get("src", "?"))] += 1
    print("=" * 60)
    print("[%s] 条数=%d  按首行书名不同=%d  按 src 不同=%d" % (key, n, len(c), len(cs)))
    print("  首行书名 Top10:")
    for k, v in c.most_common(10):
        print("    %5d  %s" % (v, k))
    print("  src Top6:")
    for k, v in cs.most_common(6):
        print("    %5d  %s" % (v, k[:50]))
    if c:
        top1 = c.most_common(1)[0][1]
        print("  ⚠️ 最大单一来源占比 = %.1f%%" % (100.0 * top1 / n))
print("DIAG_DONE")

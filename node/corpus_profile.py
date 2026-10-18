#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""corpus_profile.py — ufw_zh 构成画像: source 分布 / score 分布 / 重复率 / 引流词命中
只抽前 N 行(默认 30 万), 不扫全量。
"""
import json
import sys
from collections import Counter

FP = "/mnt/workspace/node1/corpus_04b/ufw_zh.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300000

SPAM = ["回复关键词", "关注公众号", "扫码", "加微信", "点击阅读原文", "长按识别",
        "点赞在看", "转载", "广告", "↓↓↓", "更多精彩"]

src = Counter()
score_bucket = Counter()
spam_hit = Counter()
hashes = set()
dup = 0
n = 0
short = 0
with open(FP, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        n += 1
        if n > N:
            break
        try:
            o = json.loads(line)
        except Exception:
            continue
        src[str(o.get("source", "?"))[:24]] += 1
        sc = o.get("score")
        if isinstance(sc, (int, float)):
            score_bucket[round(float(sc), 1)] += 1
        t = o.get("text", "") or ""
        if len(t) < 200:
            short += 1
        for w in SPAM:
            if w in t:
                spam_hit[w] += 1
        h = hash(t[:400])
        if h in hashes:
            dup += 1
        else:
            hashes.add(h)

print(f"抽样 {n} 行")
print(f"[source top8] {src.most_common(8)}")
print(f"[score 分布] {sorted(score_bucket.items())[:12]}")
print(f"[正文<200字] {short}  ({short/max(1,n)*100:.1f}%)")
print(f"[前400字重复] {dup}  ({dup/max(1,n)*100:.1f}%)")
print(f"[引流词命中] {spam_hit.most_common(10)}")

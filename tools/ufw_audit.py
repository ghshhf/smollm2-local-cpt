#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ufw_audit.py — 查 ufw_zh 这类"网页清洗语料"到底有多重复。

四层口径：
  L1 全文精确重复（去空白后 md5）
  L2 前缀重复（前 60 字相同）—— 命中"同一篇文章被切成多段/多站转载"
  L3 段落级转载（同一段 ≥50 字在 ≥N 篇里出现）—— 命中"正文一样、壳不同"
  L4 近重复（simhash 64bit，抽样，汉明距离 ≤3）

用法：
  python tools/ufw_audit.py --file corpus_04b/ufw_zh.jsonl [--sample-para 150000] [--near 20000]
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter

WS = re.compile(r"\s+")
PARA = re.compile(r"[\n\r]+")

# 网页噪声壳（转载站/公众号的固定话术）
NOISE = [
    "点击上方", "扫码关注", "长按识别", "微信公众号", "本文来自", "责任编辑",
    "原标题", "阅读原文", "版权归", "如有侵权", "更多精彩", "关注我们",
    "上一篇", "下一篇", "相关阅读", "免责声明", "投稿邮箱",
]


def norm(t):
    return WS.sub("", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--sample-para", type=int, default=150000, help="段落审计抽样篇数(0=全量)")
    ap.add_argument("--near", type=int, default=20000, help="近重复抽样篇数(0=跳过)")
    a = ap.parse_args()

    if not os.path.isfile(a.file):
        sys.exit("no file: %s" % a.file)

    total = os.path.getsize(a.file)
    t0 = time.time()
    seen_full = {}
    seen_pre = {}
    lens = []
    chars = 0
    n = 0
    bad_json = 0
    empty = 0
    noise_hit = 0
    trad_hit = 0
    dup_examples = []
    pre_dup_examples = []
    para_docs = {}       # para_hash -> doc 数
    para_done_docs = 0
    use_para = a.sample_para == 0
    para_limit = a.sample_para
    near_recs = []
    field_cnt = Counter()

    pairs = [("说", "說"), ("为", "為"), ("来", "來"), ("与", "與"), ("无", "無"),
             ("见", "見"), ("个", "個"), ("这", "這"), ("时", "時"), ("国", "國")]

    with open(a.file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            try:
                o = json.loads(line)
            except Exception:
                bad_json += 1
                continue
            field_cnt.update(o.keys())
            t = o.get("text") or o.get("content") or ""
            if not t.strip():
                empty += 1
                continue
            chars += len(t)
            lens.append(len(t))
            nm = norm(t)
            h = hashlib.md5(nm.encode("utf-8", "ignore")).hexdigest()
            if h in seen_full:
                seen_full[h] += 1
                if len(dup_examples) < 12:
                    dup_examples.append((seen_full[h], nm[:70]))
            else:
                seen_full[h] = 1

            p = nm[:60]
            if p in seen_pre:
                seen_pre[p] += 1
                if len(pre_dup_examples) < 12:
                    pre_dup_examples.append((seen_pre[p], p[:60]))
            else:
                seen_pre[p] = 1

            if any(x in t for x in NOISE):
                noise_hit += 1
            s = sum(t.count(x) for x, _ in pairs)
            tr = sum(t.count(y) for _, y in pairs)
            if tr > s:
                trad_hit += 1

            if a.near and len(near_recs) < a.near and random.random() < 0.02:
                near_recs.append(nm[:1500])

            if para_limit == 0 or para_done_docs < para_limit:
                if random.random() < (para_limit / max(1, 972767)) or use_para:
                    para_done_docs += 1
                    for seg in PARA.split(t):
                        seg = norm(seg)
                        if len(seg) < 50:
                            continue
                        ph = hashlib.md5(seg[:400].encode("utf-8", "ignore")).hexdigest()[:8]
                        para_docs[ph] = para_docs.get(ph, 0) + 1

            if n % 100000 == 0:
                print("  ...%d 篇  %.0f%%  %.0fs" % (n, n * 1.0 / 972767 * 100, time.time() - t0), flush=True)

    dup_docs = sum(v - 1 for v in seen_full.values() if v > 1)
    uniq = len(seen_full)
    pre_dup = sum(v - 1 for v in seen_pre.values() if v > 1)
    para_rep = sum(1 for v in para_docs.values() if v >= 3)
    para_rep_docs = sum(v for v in para_docs.values() if v >= 3)

    lens.sort()
    def pct(q):
        return lens[min(len(lens) - 1, int(len(lens) * q))] if lens else 0

    out = {
        "file": a.file,
        "bytes": total,
        "docs": n,
        "unique_full": uniq,
        "exact_dup_docs": dup_docs,
        "exact_dup_rate": round(dup_docs / max(1, n - empty), 4),
        "prefix_dup_docs": pre_dup,
        "prefix_dup_rate": round(pre_dup / max(1, n - empty), 4),
        "avg_chars": round(chars / max(1, n - empty), 1),
        "len_p50": pct(0.5), "len_p90": pct(0.9), "len_p99": pct(0.99),
        "noise_shell_docs": noise_hit,
        "noise_shell_rate": round(noise_hit / max(1, n - empty), 4),
        "trad_dominant_docs": trad_hit,
        "trad_dominant_rate": round(trad_hit / max(1, n - empty), 4),
        "para_sampled_docs": para_done_docs,
        "para_repeated_ge3": para_rep,
        "para_repeated_doc_hits": para_rep_docs,
        "bad_json": bad_json,
        "empty": empty,
        "fields": dict(field_cnt.most_common(8)),
        "top_dup_examples": dup_examples,
        "top_prefix_examples": pre_dup_examples,
        "elapsed_s": round(time.time() - t0, 1),
    }

    # L4 近重复（simhash 64bit）
    if a.near and near_recs:
        def simhash(s):
            v = [0] * 64
            for i in range(0, max(1, len(s) - 4), 3):
                g = s[i:i + 5]
                hv = int(hashlib.md5(g.encode("utf-8", "ignore")).hexdigest()[:16], 16)
                for b in range(64):
                    v[b] += 1 if (hv >> b) & 1 else -1
            return sum(1 << b for b in range(64) if v[b] > 0)

        hs = [simhash(x) for x in near_recs]
        # LSH banding：2 段 32bit（假阳率 ~2e-9）。
        # ⚠️ 旧版用 4 段 16bit，2 万篇就把 65536 个桶填满 26%，必然假碰撞，
        #    曾误报 41.9% 近重复 —— 别再用小桶。
        b0, b1 = {}, {}
        cand = 0
        near3 = 0
        for i, h in enumerate(hs):
            k0, k1 = h & 0xFFFFFFFF, (h >> 32) & 0xFFFFFFFF
            c = b0.get(k0, b1.get(k1))
            if c is not None:
                cand += 1
                if bin(h ^ hs[c]).count("1") <= 3:
                    near3 += 1
            b0.setdefault(k0, i)
            b1.setdefault(k1, i)
        out["near_sampled"] = len(hs)
        out["near_candidates"] = cand
        out["near_dup_docs"] = near3
        out["near_dup_rate"] = round(near3 / max(1, len(hs)), 4)

    dst = os.path.splitext(a.file)[0] + "_audit.json"
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print()
    print("=== 结果 ===")
    for k in ("docs", "unique_full", "exact_dup_docs", "exact_dup_rate", "prefix_dup_docs",
              "prefix_dup_rate", "avg_chars", "len_p50", "len_p90", "noise_shell_rate",
              "trad_dominant_rate", "para_repeated_ge3", "para_repeated_doc_hits",
              "near_dup_rate", "elapsed_s"):
        if k in out:
            print("  %-26s %s" % (k, out[k]))
    print("  → %s" % dst)


if __name__ == "__main__":
    main()

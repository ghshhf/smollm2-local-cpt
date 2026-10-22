#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bank_profile.py — 银行产物能力画像

回答一个具体问题：这个 0.5B + LoRA 到底「强在哪」。
方法：按任务是否属于「波类」切两半，各自看新规则下的存活率。
      若波类近乎全过、非波类大面积套壳 → 「能力」其实是 sin 模板复读，不是编程。

用法: python3 tools/bank_profile.py [--db data/triple_bank.db]
"""
import argparse
import importlib.util
import os
import sqlite3
from collections import Counter

WAVE_KW = ("正弦", "sin", "余弦", "波", "谐波", "傅里叶", "拍频", "驻波",
           "调频", "调幅", "干涉", "多普勒", "折射", "色散", "衰减")


def is_wave(task):
    t = (task or "").lower()
    return any(w in t for w in WAVE_KW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/triple_bank.db")
    ap.add_argument("--tv", default="tools/task_verify.py")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("tv", a.tv)
    tv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tv)

    c = sqlite3.connect("file:%s?mode=ro" % a.db, uri=True)
    rows = list(c.execute("SELECT task, code_text FROM triples"))

    stat = Counter()
    fail_reason = Counter()
    for task, code in rows:
        kind = "波类" if is_wave(task) else "非波类"
        r = tv.verify(task, code)
        ok = r["pass"]
        stat[(kind, "pass" if ok else "fail")] += 1
        if not ok:
            for ch in r["checks"]:
                if not ch["pass"]:
                    fail_reason[(kind, ch["name"])] += 1

    print("=== 银行产物能力画像（新规则）===")
    for kind in ("波类", "非波类"):
        p = stat[(kind, "pass")]
        f = stat[(kind, "fail")]
        n = p + f
        if n:
            print("%s: %d 条 | 存活 %d (%.1f%%) | 被拦 %d (%.1f%%)"
                  % (kind, n, p, p * 100.0 / n, f, f * 100.0 / n))
    print("--- 被拦归因 ---")
    for (kind, name), v in fail_reason.most_common(12):
        print("  %-4s %-14s %d" % (kind, name, v))


if __name__ == "__main__":
    main()

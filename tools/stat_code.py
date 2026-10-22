#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/stat_code.py — 统计代码语料真实规模（不猜，直接读 parquet）

用法: python3 tools/stat_code.py [目录]
口径: 字符数 / 估算 token（代码约 3.2 字节/token，中文约 1.5 字/token，取混合 3.0）
"""
import glob
import os
import sys

import pyarrow.parquet as pq

D = sys.argv[1] if len(sys.argv) > 1 else "/mnt/workspace/node1/data/code_raw/smolpyedu"
files = sorted(glob.glob(os.path.join(D, "*.parquet")))
tot_rows = 0
tot_chars = 0
extra_cols = set()
samples = []

for f in files:
    if os.path.basename(f).startswith("."):
        continue
    try:
        t = pq.read_table(f)
    except Exception as e:
        print("  [skip] %s %s" % (os.path.basename(f), e))
        continue
    names = t.column_names
    col = "content" if "content" in names else names[0]
    extra_cols.update(names)
    n = t.num_rows
    chars = 0
    for i, v in enumerate(t.column(col).to_pylist()):
        s = v if isinstance(v, str) else str(v)
        chars += len(s)
        if len(samples) < 3 and n > 100:
            samples.append(s[:160].replace("\n", "\\n"))
    tot_rows += n
    tot_chars += chars
    print("  %-34s %7d 行  %10.1f MB" % (os.path.basename(f), n, chars / 1e6))

print("\n合计 %d 行 / %.2f 亿字符 / 列=%s" % (tot_rows, tot_chars / 1e8, sorted(extra_cols)))
for t in (3.0, 3.5, 4.0):
    print("  按 %.1f 字节/token ⇒ %.2f 亿 token" % (t, tot_chars / t / 1e8))
print("\n样本:")
for s in samples:
    print("  |", s[:150])

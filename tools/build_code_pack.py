#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/build_code_pack.py — 从 SmolPyEdu 抽定量的裸 Python 代码包

用户口径（2026-10-19）：代码取现成，不顶格。目标 1.5 亿 token，
按 3.5 字节/token 折算 = 5.25 亿字符；留去重/截断冗余，默认抽 6 亿字符。

均匀抽样：10 片平均取，避免只吃某一片的领域偏置。

用法:
    python3 tools/build_code_pack.py                    # 默认 6 亿字符
    python3 tools/build_code_pack.py --chars 5.25e8
输出:
    /mnt/workspace/node1/corpus_04b/code_pretrain.jsonl   {"text": ...}
"""
import argparse
import glob
import json
import os
import sys

import pyarrow.parquet as pq

SRC = "/mnt/workspace/node1/data/code_raw/smolpyedu"
OUT = "/mnt/workspace/node1/corpus_04b/code_pretrain.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--chars", type=float, default=6.0e8, help="目标字符数")
    ap.add_argument("--min-len", type=int, default=200, help="单条最少字符（太短的片段信息量低）")
    ap.add_argument("--max-len", type=int, default=20000, help="单条上限，超了跳过（防单文件刷量）")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.src, "*.parquet"))
                   if not os.path.basename(f).startswith("."))
    if not files:
        print("找不到 parquet: %s" % a.src)
        return 1
    per = a.chars / len(files)
    print("源 %d 片，目标共 %.2f 亿字符（每片 %.2f 亿）" % (len(files), a.chars / 1e8, per / 1e8))

    tmp = a.out + ".tmp"
    n_rows = 0
    n_chars = 0
    n_skip = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for fi, path in enumerate(files):
            t = pq.read_table(path, columns=["text"])
            col = t.column("text").to_pylist()
            got = 0
            kept = 0
            for s in col:
                if not isinstance(s, str):
                    n_skip += 1
                    continue
                L = len(s)
                if L < a.min_len or L > a.max_len:
                    n_skip += 1
                    continue
                f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
                got += L
                kept += 1
                if got >= per:
                    break
            n_rows += kept
            n_chars += got
            print("  片%02d 取 %6d 行 / %.2f 亿字符（源 %d 行）"
                  % (fi, kept, got / 1e8, len(col)), flush=True)
    os.replace(tmp, a.out)
    print("\n完成 %d 行 / %.2f 亿字符 / 跳过 %d 条" % (n_rows, n_chars / 1e8, n_skip))
    print("按 3.5 字节/token ⇒ 约 %.2f 亿 token" % (n_chars / 3.5 / 1e8))
    print("输出: %s (%.2f GB)" % (a.out, os.path.getsize(a.out) / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())

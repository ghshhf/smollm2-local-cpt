#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""peek_corpus.py — 体检 0.4B 正在训练的真实语料
输出: 每份文件的大小/行数/字段/抽样原文(截断), 以及合并文件的配比。
用法: python3 peek_corpus.py [每份抽几条=3]
"""
import json
import os
import sys

W = "/mnt/workspace/node1"
SRCS = [
    ("ufw_zh 中文通用", f"{W}/corpus_04b/ufw_zh.jsonl"),
    ("创意文章(小说)", f"{W}/corpus_04b/creative_articles_novel.jsonl"),
    ("古诗词", f"{W}/corpus_04b/poetry_zh.jsonl"),
    ("owner 自有资料", "/root/owner_corpus.jsonl"),
    ("合并产物(实际喂的)", "/root/mix_04b.jsonl"),
]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
PREV = 260


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


for label, fp in SRCS:
    print("=" * 70)
    if not os.path.exists(fp):
        print(f"## {label}  缺失 {fp}")
        continue
    sz = os.path.getsize(fp)
    print(f"## {label}")
    print(f"   path={fp}  size={human(sz)}")
    keys = {}
    shown = 0
    nline = 0
    chars = 0
    with open(fp, "rb") as f:
        for raw in f:
            nline += 1
            s = raw.decode("utf-8", "ignore").strip()
            if not s:
                continue
            try:
                o = json.loads(s)
                if isinstance(o, dict):
                    for k in o:
                        keys[k] = keys.get(k, 0) + 1
                    txt = o.get("text") or o.get("content") or o.get("completion") or ""
                else:
                    txt = s
            except Exception:
                txt = s
            chars += len(txt)
            if shown < N and len(txt) > 60:
                shown += 1
                flat = " ".join(txt.split())
                print(f"   [{shown}] {flat[:PREV]}{'...' if len(flat) > PREV else ''}")
    print(f"   lines={nline}  chars={chars}  chars/line={chars / max(1, nline):.0f}")
    print(f"   keys={dict(list(keys.items())[:6])}")

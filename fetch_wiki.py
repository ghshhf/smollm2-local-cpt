#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把下载到的中文维基区间文件 (_wiki_chunk.json) 解析成训练用 jsonl。
- 每行是 {id,title,tags,text} 的 JSON
- 提取 title + text, 写成 pretrain_corpus/_wiki_tmp.jsonl {"text":..., "src":"wiki"}
- 跳过过短 / 解析失败的残行(区间下载可能截断末行)
用法:
  python fetch_wiki.py
"""
import os, json

PC = "D:/work/_local_ai/train_15b/pretrain_corpus"
CHUNK = os.path.join(PC, "_wiki_chunk.json")
OUT = os.path.join(PC, "_wiki_tmp.jsonl")

MIN = 30
n = 0
bad = 0
with open(CHUNK, encoding="utf-8", errors="ignore") as f, \
     open(OUT, "w", encoding="utf-8") as g:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            bad += 1
            continue
        if not isinstance(d, dict):
            bad += 1
            continue
        text = (d.get("text") or "").strip()
        title = (d.get("title") or "").strip()
        if len(text) < MIN:
            continue
        body = (title + "\n" + text) if title else text
        g.write(json.dumps({"text": body, "src": "wiki"}, ensure_ascii=False) + "\n")
        n += 1
print(f"维基解析完成: 保留 {n:,} 条, 跳过残行 {bad:,}")
print(f"  输出: {OUT}")

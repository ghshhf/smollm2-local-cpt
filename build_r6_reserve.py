#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_r6_reserve.py — 产出 R6 知识封顶语料 zh_text_r6.jsonl (0.2B 级知识天花板 + 0.3B 知识储备种子)
策略: 以 zh_text_know_full.jsonl(百科+教育, 255MB) 为知识锚, 再并入:
  - zh_text_v2.jsonl      前 50% 作为 散文/通用/对话层 (cat=prose)
  - zh_text_next_dedup.jsonl 前 20% 作为 社科/杂谈层 (cat=social)
全局去重(md5 of 空白归一化文本)。AI 池(zh_text_v2_ai) 严格不进。
用法: python build_r6_reserve.py
输出: pretrain_corpus/zh_text_r6.jsonl  (chain 的 stage_corpus R6 找 zh_text_r6.jsonl)
后续增长(0.3B 储备 -> 600MB~1GB+): 加 zhwiki 全量 / 古籍诸子 / 科普, 见 grow_r6_reserve.py(待下载源)。
"""
import os, re, json, hashlib

ROOT = "D:/work/_local_ai/train_15b"
PC = os.path.join(ROOT, "pretrain_corpus")

BASE   = os.path.join(PC, "zh_text_know_full.jsonl")   # 知识锚 (v3_edu + wiki)
PROSE  = os.path.join(PC, "zh_text_v2.jsonl")          # 通用中文 (散文/对话)
SOCIAL = os.path.join(PC, "zh_text_next_dedup.jsonl")   # 网页 (社科/杂谈)
OUT    = os.path.join(PC, "zh_text_r6.jsonl")

PROSE_FRAC  = 0.50   # 取前 50%
SOCIAL_FRAC = 0.20   # 取前 20%

ws_re = re.compile(r'\s+')
MIN_LEN = 20

def extract_text(line):
    line = line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
        if isinstance(d, dict):
            t = d.get("text") or d.get("output") or d.get("content") or ""
        else:
            t = str(d)
    except Exception:
        t = line
    t = t.strip()
    if len(t) < MIN_LEN:
        return None
    return t

def counted_lines(fp):
    n = 0
    with open(fp, encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n

def main():
    seen = set()
    n = c = 0
    comp = {}

    def emit(text, src, cat):
        nonlocal n, c
        h = hashlib.md5(ws_re.sub('', text).encode("utf-8")).hexdigest()
        if h in seen:
            return
        seen.add(h)
        with open(OUT, "a", encoding="utf-8") as fo:
            fo.write(json.dumps({"text": text, "src": src, "cat": cat}, ensure_ascii=False) + "\n")
        n += 1; c += len(text)
        comp[cat] = comp.get(cat, 0) + 1

    # 先清空输出
    open(OUT, "w", encoding="utf-8").close()

    # 1) 知识锚: 全量保留
    print("[1/3] 知识锚 know_full ...")
    cnt = 0
    for line in open(BASE, encoding="utf-8", errors="ignore"):
        t = extract_text(line)
        if t is None:
            continue
        # 尽量保留原 src/cat, 缺则补 knowledge
        try:
            d = json.loads(line)
            src = d.get("src", "know")
            cat = d.get("cat", "knowledge")
        except Exception:
            src, cat = "know", "knowledge"
        emit(t, src, cat); cnt += 1
    print(f"     保留 {cnt:,} 篇")

    # 2) 散文/通用层: v2 前 PROSE_FRAC
    total = counted_lines(PROSE)
    take = int(total * PROSE_FRAC)
    print(f"[2/3] 散文/通用 v2 前 {PROSE_FRAC:.0%} (取 {take:,}/{total:,}) ...")
    cnt = 0
    with open(PROSE, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= take:
                break
            t = extract_text(line)
            if t is None:
                continue
            emit(t, "v2", "prose"); cnt += 1
    print(f"     保留 {cnt:,} 篇")

    # 3) 社科/杂谈层: next_dedup 前 SOCIAL_FRAC
    total = counted_lines(SOCIAL)
    take = int(total * SOCIAL_FRAC)
    print(f"[3/3] 社科/杂谈 next_dedup 前 {SOCIAL_FRAC:.0%} (取 {take:,}/{total:,}) ...")
    cnt = 0
    with open(SOCIAL, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= take:
                break
            t = extract_text(line)
            if t is None:
                continue
            emit(t, "next", "social"); cnt += 1
    print(f"     保留 {cnt:,} 篇")

    size = os.path.getsize(OUT)
    print(f"\n-> zh_text_r6.jsonl: {n:,} 篇 / ~{c/1.6:,.0f} token / {size/1e6:.1f} MB")
    print("   构成(按 cat):")
    for k, v in sorted(comp.items(), key=lambda x: -x[1]):
        print(f"     {k}: {v:,}")

if __name__ == "__main__":
    main()

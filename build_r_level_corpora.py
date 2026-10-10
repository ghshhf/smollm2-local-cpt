#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重建分级知识语料(用于 R2 之后按容量逐级扩容的链):
  - zh_text_lang.jsonl     (cat=lang)     : v2(通用中文) + next_dedup(网页)  -> L2 语言基础 R4
  - zh_text_know_edu.jsonl (cat=knowledge) : v3_edu(教育)                    -> L2 知识·教育 R5(封顶)
  - zh_text_know_full.jsonl(cat=knowledge) : v3_edu + _wiki_tmp(维基) 全量    -> L3 全量知识 R6
全局去重(三段之间也去重)。AI 池(zh_text_v2_ai) 严格不进。
用法: python build_r_level_corpora.py
"""
import os, re, json, hashlib

ROOT = "D:/work/_local_ai/train_15b"
PC = os.path.join(ROOT, "pretrain_corpus")
OUT_LANG = os.path.join(PC, "zh_text_lang.jsonl")
OUT_EDU  = os.path.join(PC, "zh_text_know_edu.jsonl")
OUT_FULL = os.path.join(PC, "zh_text_know_full.jsonl")

# 每个输出文件独立去重(允许 v3_edu 同时进 R5 与 R6: L2 训片段, L3 训全量)
TARGETS = [
    (OUT_LANG, "lang", [("v2", os.path.join(PC, "zh_text_v2.jsonl")),
                        ("next_dedup", os.path.join(PC, "zh_text_next_dedup.jsonl"))]),
    (OUT_EDU,  "knowledge", [("v3_edu", os.path.join(PC, "zh_text_v3_edu.jsonl"))]),
    (OUT_FULL, "knowledge", [("v3_edu", os.path.join(PC, "zh_text_v3_edu.jsonl")),
                              ("wiki", os.path.join(PC, "_wiki_tmp.jsonl"))]),
]
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
    if not t:
        return None
    t = t.strip()
    if len(t) < MIN_LEN:
        return None
    return t

def main():
    for out_path, cat, srcs in TARGETS:
        seen = set()
        n = c = 0
        with open(out_path, "w", encoding="utf-8") as fo:
            for label, fp in srcs:
                if not os.path.isfile(fp):
                    print(f"  [跳过] {label}: 不存在 ({fp})")
                    continue
                cnt = 0
                for line in open(fp, encoding="utf-8", errors="ignore"):
                    t = extract_text(line)
                    if t is None:
                        continue
                    h = hashlib.md5(ws_re.sub('', t).encode("utf-8")).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    fo.write(json.dumps({"text": t, "src": label, "cat": cat}, ensure_ascii=False) + "\n")
                    cnt += 1; n += 1; c += len(t)
                print(f"  [{label}->{cat}] 保留 {cnt:,} 篇")
        tag = os.path.basename(out_path)
        print(f"  -> {tag}: {n:,} 篇 / ~{c/1.6:,.0f} token\n")

if __name__ == "__main__":
    main()

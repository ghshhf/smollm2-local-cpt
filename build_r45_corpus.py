#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 R3 混合知识语料按「知识分类」切成两段, 对应 R4 / R5 (在 0.2B 模型上训):
  - R4 语言基础类 (cat=lang):   zh_text_v2(通用中文) + zh_text_next_dedup(通用网页)
  - R5 知识类    (cat=knowledge): zh_text_v3_edu(教育/百科) + _wiki_tmp(维基百科)
每段 1 epoch; 两段合计 ≈ R3 单 epoch 步数(把 15 天整块切成两个有里程碑的短段)。
AI 池(zh_text_v2_ai) 严格不进。
用法: python build_r45_corpus.py
"""
import os, re, json, hashlib

ROOT = "D:/work/_local_ai/train_15b"
PC = os.path.join(ROOT, "pretrain_corpus")
OUT_R4 = os.path.join(PC, "zh_text_r4.jsonl")
OUT_R5 = os.path.join(PC, "zh_text_r5.jsonl")

# (标签, 来源文件, 目标 corpus, 分类)
GROUPS = [
    ("v2",        os.path.join(PC, "zh_text_v2.jsonl"),         OUT_R4, "lang"),
    ("next_dedup",os.path.join(PC, "zh_text_next_dedup.jsonl"), OUT_R4, "lang"),
    ("v3_edu",    os.path.join(PC, "zh_text_v3_edu.jsonl"),     OUT_R5, "knowledge"),
    ("wiki",      os.path.join(PC, "_wiki_tmp.jsonl"),          OUT_R5, "knowledge"),
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
    # 全局去重(两段之间也去重, 避免同一篇在 R4/R5 重复)
    seen = set()
    n_r4 = n_r5 = 0
    chars_r4 = chars_r5 = 0
    with open(OUT_R4, "w", encoding="utf-8") as f4, open(OUT_R5, "w", encoding="utf-8") as f5:
        for label, fp, out_path, cat in GROUPS:
            if not os.path.isfile(fp):
                print(f"  [跳过] {label}: 不存在 ({fp})")
                continue
            cnt = 0
            fout = f4 if out_path == OUT_R4 else f5
            for line in open(fp, encoding="utf-8", errors="ignore"):
                t = extract_text(line)
                if t is None:
                    continue
                h = hashlib.md5(ws_re.sub('', t).encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                rec = json.dumps({"text": t, "src": label, "cat": cat}, ensure_ascii=False)
                fout.write(rec + "\n")
                cnt += 1
                if cat == "lang":
                    n_r4 += 1; chars_r4 += len(t)
                else:
                    n_r5 += 1; chars_r5 += len(t)
            print(f"  [{label}->{cat}] 保留 {cnt:,} 篇")
    print(f"\nR4 语言基础: {n_r4:,} 篇 / {chars_r4:,} 字符 / ~{chars_r4/1.6:,.0f} token  -> {OUT_R4}")
    print(f"R5 知识:     {n_r5:,} 篇 / {chars_r5:,} 字符 / ~{chars_r5/1.6:,.0f} token  -> {OUT_R5}")
    print(f"两段合计 ~{(chars_r4+chars_r5)/1.6:,.0f} token (≈ R3 单 epoch)")

if __name__ == "__main__":
    main()

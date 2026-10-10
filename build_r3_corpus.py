#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 R3 混合知识语料 (pretrain_corpus/zh_text_r3.jsonl)
组成:
  - zh_text_v2.jsonl           (整份, 覆盖 R1 漏训的 ~35% + 复训已见 65%, 通用中文)
  - zh_text_v3_edu.jsonl       (教育/百科类知识, 本地已趴着没用)
  - zh_text_next_dedup.jsonl   (去重后的通用网页文本)
  - _wiki_tmp.jsonl            (可选, 若已下载中文维基子集则并入)
全局精确去重 (md5), 排除过短/空行。AI 池(zh_text_v2_ai) 严格不进。
用法:
  python build_r3_corpus.py
"""
import os, re, json, hashlib

ROOT = "D:/work/_local_ai/train_15b"
PC = os.path.join(ROOT, "pretrain_corpus")
OUT = os.path.join(PC, "zh_text_r3.jsonl")

SOURCES = [
    ("v2",        os.path.join(PC, "zh_text_v2.jsonl")),
    ("v3_edu",    os.path.join(PC, "zh_text_v3_edu.jsonl")),
    ("next_dedup",os.path.join(PC, "zh_text_next_dedup.jsonl")),
    ("wiki",      os.path.join(PC, "_wiki_tmp.jsonl")),   # 可选
]
ws_re = re.compile(r'\s+')
cn = re.compile(r'[\u4e00-\u9fff]')
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
    seen = set()
    n_total = 0
    n_kept = 0
    chars_total = 0
    per = {}
    with open(OUT, "w", encoding="utf-8") as fout:
        for label, fp in SOURCES:
            if not os.path.isfile(fp):
                print(f"  [跳过] {label}: 文件不存在 ({fp})")
                continue
            cnt = 0
            for line in open(fp, encoding="utf-8", errors="ignore"):
                t = extract_text(line)
                if t is None:
                    continue
                n_total += 1
                chars_total += len(t)
                h = hashlib.md5(ws_re.sub('', t).encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                fout.write(json.dumps({"text": t, "src": label}, ensure_ascii=False) + "\n")
                n_kept += 1
                cnt += 1
            per[label] = cnt
            print(f"  [{label}] 去重后保留 {cnt:,} 篇")
    print(f"\nR3 语料: {n_kept:,} 篇 (原始 {n_total:,}, 去重丢弃 {n_total - n_kept:,})")
    print(f"  总字符 {chars_total:,}  估算 token {chars_total/1.6:,.0f}  (中文 ~1.6 字/token)")
    print(f"  输出: {OUT}")

if __name__ == "__main__":
    main()

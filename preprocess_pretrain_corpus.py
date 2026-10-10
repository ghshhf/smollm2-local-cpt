#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 data/ 下的指令数据预处理成纯文本预训练语料（供 0.1B 级全量/持续预训练线使用）。

策略：
- 每条取 instruction + output 拼成一篇文档（保留"前文→续写"上下文，适合 continual pretrain）
- 过滤 output 过短(<50字)的噪声样本
- 全文精确去重
- 输出 jsonl of {"text": doc}，兼容 HF datasets.load_dataset("json")
- 可复跑：资料新增后重跑即可刷新
"""
import json, glob, os, collections

OUT_DIR = "pretrain_corpus"
OUT_FILE = os.path.join(OUT_DIR, "zh_text.jsonl")
MIN_OUT = 50  # output 低于此字符数跳过

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    docs = []
    seen = set()
    per_src = collections.Counter()
    skipped_short = 0
    skipped_dup = 0
    for f in sorted(glob.glob("data/**/*.jsonl", recursive=True)):
        try:
            rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        except Exception as e:
            print(f"  [跳过] {f}: {e}")
            continue
        for r in rows:
            ins = (r.get("instruction") or "").strip()
            out = (r.get("output") or "").strip()
            if len(out) < MIN_OUT:
                skipped_short += 1
                continue
            doc = (ins + "\n" + out) if ins else out
            key = doc
            if key in seen:
                skipped_dup += 1
                continue
            seen.add(key)
            docs.append(doc)
            per_src[os.path.basename(f)] += 1
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    total_chars = sum(len(d) for d in docs)
    print(f"=== 预处理完成 ===")
    print(f"输出: {OUT_FILE}")
    print(f"文档数: {len(docs):,}")
    print(f"总字符: {total_chars:,}  | 估算 token(0.65/字): ~{int(total_chars*0.65):,}")
    print(f"跳过(过短<{MIN_OUT}字): {skipped_short:,} | 跳过(精确重复): {skipped_dup:,}")
    print(f"\n=== 各来源文档数(top) ===")
    for name, n in per_src.most_common(12):
        print(f"  {n:>6,}  {name}")

if __name__ == "__main__":
    main()

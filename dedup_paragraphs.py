# -*- coding: utf-8 -*-
"""第二道去重：段落级。删除同一篇内重复出现的段落(水印/重复推送行)。
用法: python dedup_paragraphs.py <in.jsonl> <out.jsonl>
"""
import json, sys, re

def main():
    fin, fout = sys.argv[1], sys.argv[2]
    docs, kept_chars, drop_chars, drop_paras = 0, 0, 0, 0
    with open(fin, "r", encoding="utf-8", errors="ignore") as fi, \
         open(fout, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if len(t) < 50:
                continue
            paras = re.split(r"\n+", t)
            seen = set()
            out = []
            for p in paras:
                key = re.sub(r"\s+", "", p)
                if len(key) < 12:          # 太短的段落不去重(对话体本就重复)
                    out.append(p)
                    continue
                if key in seen:
                    drop_paras += 1
                    drop_chars += len(key)
                    continue
                seen.add(key)
                out.append(p)
            nt = "\n".join(out)
            if len(nt) < 50:
                continue
            fo.write(json.dumps({"text": nt}, ensure_ascii=False) + "\n")
            docs += 1
            kept_chars += len(nt)
    print(f"文档 {docs}；保留 {kept_chars} 字；剔除重复段落 {drop_paras} 段 / {drop_chars} 字 "
          f"({drop_chars/max(1,kept_chars+drop_chars)*100:.2f}%)")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""语料去重：精确重复 + 转载近似重复(首段指纹+长度相近) 两道过滤。
用法: python dedup_corpus.py <in.jsonl> <out.jsonl>
"""
import json, sys, hashlib, re

def norm(s):
    s = re.sub(r"\s+", "", s)
    return s

def main():
    fin, fout = sys.argv[1], sys.argv[2]
    seen_full = set()
    seen_fp = {}          # (首段指纹) -> 长度集合
    kept, drop_full, drop_fp = 0, 0, 0
    chars = 0
    with open(fin, "r", encoding="utf-8", errors="ignore") as fi, \
         open(fout, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if len(t) < 50:
                continue
            n = norm(t)
            h = hashlib.md5(n.encode("utf-8")).hexdigest()
            if h in seen_full:
                drop_full += 1
                continue
            # 转载近似: 前 300 字指纹 + 长度相差 <8%
            fp = hashlib.md5(n[:300].encode("utf-8")).hexdigest()
            L = len(n)
            dup = False
            for prev_L in seen_fp.get(fp, ()):
                if abs(prev_L - L) / max(1, prev_L) < 0.08:
                    dup = True
                    break
            if dup:
                drop_fp += 1
                continue
            seen_full.add(h)
            seen_fp.setdefault(fp, []).append(L)
            fo.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            kept += 1
            chars += len(n)
    print(f"保留 {kept} 篇 / {chars} 字；精确重复剔除 {drop_full}；近似转载剔除 {drop_fp}")

if __name__ == "__main__":
    main()

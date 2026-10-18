#!/usr/bin/env python3
# 从 stdin 读 unzip -p 解出的 articles.json(JSONL 流), 切片成 zh_r_weixin.jsonl
import sys, json
OUT = r"D:\work\_local_ai\train_15b\pretrain_corpus\zh_r_weixin.jsonl"
CAP = 320e6
n = 0; tot = 0
with open(OUT, "w", encoding="utf-8") as w:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        c = o.get("content") or ""
        t = o.get("title") or ""
        nm = o.get("name") or o.get("account") or "?"
        if len(c) < 80:
            continue
        rec = {"text": (t + "\n" + c).strip(), "src": "weixin:" + str(nm), "cat": "modern"}
        s = json.dumps(rec, ensure_ascii=False) + "\n"
        w.write(s); tot += len(s.encode("utf-8")); n += 1
        if tot >= CAP:
            break
print(f"[done] weixin lines={n} bytes={tot} -> {OUT}")

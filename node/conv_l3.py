#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""conv_l3.py — node1 上把 Ultra-FineWeb-L3 zh parquet 转 jsonl（蒸馏补充，单独标记可回滚）。"""
import glob, json, os, sys, time

D = "/mnt/workspace/node1/corpus_04b"

def L(m):
    print(m, flush=True)

try:
    import pyarrow.parquet as pq
except ImportError:
    L("[cv] install pyarrow ...")
    os.system("%s -m pip install -q pyarrow" % sys.executable)
    import pyarrow.parquet as pq

def extract(r):
    c = r.get("content") or r.get("text") or ""
    if isinstance(c, str) and c.strip():
        return c.strip()
    q = (r.get("question") or "").strip()
    a = (r.get("answer") or r.get("output") or "").strip()
    if q and a:
        return q + "\n" + a
    parts = [v.strip() for k, v in r.items()
             if isinstance(v, str) and len(v.strip()) > 40
             and k not in ("style", "source", "id", "lang")]
    return "\n".join(parts)

for sub, out in (("l3_ms", "l3_multistyle.jsonl"), ("l3_qa", "l3_qa.jsonl")):
    ps = sorted(glob.glob(os.path.join(D, sub, "*.parquet")))
    L("[cv] %s: %d files" % (sub, len(ps)))
    n = tot = 0
    with open(os.path.join(D, out), "w", encoding="utf-8") as f:
        for p in ps:
            try:
                pf = pq.ParquetFile(p)
            except Exception as e:
                L("[cv] SKIP_BAD %s %s" % (os.path.basename(p), str(e)[:80]))
                continue
            L("[cv] schema: %s" % pf.schema_arrow.names)
            cols = [c for c in ("content", "text", "question", "answer",
                                "output", "style", "source") if c in pf.schema_arrow.names]
            for b in pf.iter_batches(batch_size=1000, columns=cols or None):
                for r in b.to_pylist():
                    t = extract(r)
                    if len(t) < 150:
                        continue
                    f.write(json.dumps({"text": t, "source": "l3_" + sub},
                                       ensure_ascii=False) + "\n")
                    n += 1
                    tot += len(t)
    L("[cv] %s DONE rows=%d chars=%.1fM -> %s"
      % (sub, n, tot / 1e6, os.path.getsize(os.path.join(D, out)) / 1e6))

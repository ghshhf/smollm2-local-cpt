#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""conv_ufw.py — 在 node1 上把 Ultra-FineWeb-zh parquet 转 jsonl（囤原始文本，供 0.4B）。"""
import glob, json, os, sys, time

D = "/mnt/workspace/node1/corpus_04b"
OUT = os.path.join(D, "ufw_zh.jsonl")
MIN_CHARS = 200

def L(m):
    print(m, flush=True)

try:
    import pyarrow.parquet as pq
except ImportError:
    L("[cv] install pyarrow ...")
    os.system("%s -m pip install -q pyarrow" % sys.executable)
    import pyarrow.parquet as pq

os.makedirs(D, exist_ok=True)
ps = sorted(glob.glob(os.path.join(D, "ufw", "*.parquet")))
L("[cv] %d parquet files" % len(ps))
n = skip = tot = 0
t0 = time.time()
with open(OUT, "w", encoding="utf-8") as f:
    for p in ps:
        try:
            pf = pq.ParquetFile(p)
        except Exception as e:
            L("[cv] SKIP_BAD %s %s" % (os.path.basename(p), str(e)[:80]))
            continue
        names = pf.schema_arrow.names
        cols = [c for c in ("content", "score", "source") if c in names]
        c0 = n
        for b in pf.iter_batches(batch_size=2000, columns=cols):
            for r in b.to_pylist():
                t = (r.get("content") or "").strip()
                if len(t) < MIN_CHARS:
                    skip += 1
                    continue
                f.write(json.dumps({"text": t, "score": r.get("score"),
                                    "source": "ufw"}, ensure_ascii=False) + "\n")
                n += 1
                tot += len(t)
        L("[cv] %s -> +%d rows (cum %d, %.0fs)"
          % (os.path.basename(p), n - c0, n, time.time() - t0))
L("[cv] DONE rows=%d skipped=%d chars=%.1fM -> %s (%.1fMB)"
  % (n, skip, tot / 1e6, OUT, os.path.getsize(OUT) / 1e6))

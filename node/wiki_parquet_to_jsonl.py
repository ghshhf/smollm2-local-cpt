#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wiki_parquet_to_jsonl.py — 把 wiki_zh_shard*.parquet 转成统一 jsonl 语料。
输出 {"text": "<title>\\n<body>", "title": ...}；跳过空正文/过短条目。
"""
import sys, os, json, io

try:
    import pyarrow.parquet as pq
except Exception as e:
    print("NO_PYARROW", e)
    sys.exit(1)

out = sys.argv[1]
minsz = int(os.environ.get("MIN_CHARS", "120"))
n = tot = skip = 0
with io.open(out, "w", encoding="utf-8") as f:
    for src in sys.argv[2:]:
        try:
            pf = pq.ParquetFile(src)
        except Exception as e:
            print("SKIP_BAD", src, str(e)[:90])
            continue
        cnt = 0
        for batch in pf.iter_batches(batch_size=2000, columns=["title", "text"]):
            for r in batch.to_pylist():
                t = (r.get("title") or "").strip()
                b = (r.get("text") or "").strip()
                if len(b) < minsz or not t:
                    skip += 1
                    continue
                f.write(json.dumps({"text": t + "\n" + b, "title": t},
                                   ensure_ascii=False) + "\n")
                cnt += 1
        print("  %s -> +%d" % (os.path.basename(src), cnt))
        n += cnt
    print("WROTE %d lines (skipped %d) -> %s (%.1fMB)"
          % (n, skip, out, os.path.getsize(out) / 1e6))
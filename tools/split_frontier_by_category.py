# -*- coding: utf-8 -*-
"""split_frontier_by_category.py — 把去重后的 frontier 论文按类物理拆分。
复用 tools/classify_frontier.py 的 RULES/classify，避免规则漂移。
输出:
  data/frontier/by_category/<类别>.jsonl   每类一篇（含原字段 + _cat/_tier）
  data/frontier/by_category/_index.json    类别 -> 条数 + 文件映射
用法: python tools/split_frontier_by_category.py
"""
import os, sys, json, glob, re, time
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from classify_frontier import RULES, classify  # noqa

OUT_DIR = os.path.join(ROOT, "data", "frontier")
CAT_DIR = os.path.join(OUT_DIR, "by_category")


def main():
    papers = OrderedDict()
    for fp in sorted(glob.glob(os.path.join(OUT_DIR, "2026-*.jsonl"))):
        for ln in open(fp, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            papers[d["id"]] = d
    for p in papers.values():
        name, tier = classify(p.get("title", "") + " " + (p.get("abstract") or "")[:600])
        p["_cat"], p["_tier"] = name, tier

    order = [r[0] for r in RULES] + ["其他"]
    buckets = {}
    for p in papers.values():
        buckets.setdefault(p["_cat"], []).append(p)

    os.makedirs(CAT_DIR, exist_ok=True)
    index = {"generated": time.strftime("%Y-%m-%d %H:%M"),
             "total": len(papers), "categories": {}}
    def safe(n):
        return re.sub(r'[\\/:*?"<>|]', "_", n)
    for name in sorted(buckets, key=lambda n: order.index(n)):
        plist = sorted(buckets[name],
                       key=lambda p: ({"core": 0, "evo": 1}.get(p["_tier"], 2), p.get("title", "")))
        fn = os.path.join(CAT_DIR, safe(name) + ".jsonl")
        with open(fn, "w", encoding="utf-8") as f:
            for p in plist:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        index["categories"][name] = {"count": len(plist),
                                     "file": os.path.relpath(fn, ROOT),
                                     "tier": next((t for t in (p["_tier"] for p in plist) if t), None)}
    with open(os.path.join(CAT_DIR, "_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("拆分完成:", len(papers), "篇 ->", CAT_DIR)
    for name in sorted(buckets, key=lambda n: order.index(n)):
        print("  %-22s %d 篇" % (name, len(buckets[name])))


if __name__ == "__main__":
    main()

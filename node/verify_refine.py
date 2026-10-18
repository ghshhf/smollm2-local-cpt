#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_refine.py — 精炼语料验收：JSON 合法性 + 规模 + 长度分布 + 内容抽样。
验收纪律：语料不能只看"条数/字节"数字放行，必须肉眼过一遍抽样的实际内容。
"""
import os, sys, json, statistics, io

D = os.environ.get("REF_DIR", "/mnt/workspace/node1/chain/corpus_reserve_refined")
CJK = re_cjk = None
import re
CJK = re.compile(r"[\u4e00-\u9fff]")


def main():
    out = []
    P = lambda m: (print(m, flush=True), out.append(m))
    for key in ["knowledge", "ancient", "novels", "weixin"]:
        p = os.path.join(D, "zh_r_%s.jsonl" % key)
        if not os.path.exists(p):
            P("[%s] !! 缺失 %s" % (key, p))
            continue
        n, bad, lens, hans = 0, 0, [], 0
        firsts = []
        with io.open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    t = d.get("text", "")
                except Exception:
                    bad += 1
                    continue
                if not isinstance(t, str) or not t:
                    bad += 1
                    continue
                n += 1
                lens.append(len(t))
                hans += len(CJK.findall(t))
                if len(firsts) < 3:
                    firsts.append(t)
        if not lens:
            P("[%s] !! 空文件" % key)
            continue
        lens.sort()
        P("=" * 70)
        P("[%s] %s" % (key, p))
        P("  条数=%d  坏行=%d  字节=%.1fMB  总汉字=%.0f万  ≈token=%.2fM"
          % (n, bad, os.path.getsize(p) / 1e6, hans / 1e4, hans * 2.4589 / 1e6))
        P("  长度: min=%d p25=%d 中位=%d p75=%d max=%d 均值=%d"
          % (lens[0], lens[len(lens) // 4], statistics.median(lens),
             lens[len(lens) * 3 // 4], lens[-1], statistics.mean(lens)))
        P("  汉字占比=%.3f" % (hans / max(1, sum(lens))))
        for i, t in enumerate(firsts):
            P("  --- 样例%d (len=%d) ---" % (i + 1, len(t)))
            P("  " + t[:280].replace("\n", " ⏎ "))
        if bad:
            P("  ⚠️ 坏行 %d 条" % bad)
    P("=" * 70)
    P("VERIFY_DONE")

    rp = "/tmp/verify_refine_out.txt"
    with io.open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()

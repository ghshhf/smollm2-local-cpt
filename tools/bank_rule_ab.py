#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bank_rule_ab.py — 打分机规则改动 A/B 回测器

用途：改 task_verify.py 的规则后，别只信 selftest（那是我自己编的样例）。
      拿真实银行全表，旧版 vs 新版逐条对比，量化：
        - newly_fail：新版新增拦下的（改革收益，但要逐条抽查是不是真该拦）
        - newly_pass：新版反而放过的（危险信号，说明规则松动）
        - 新拦的 check 归因分布

用法:
  python3 tools/bank_rule_ab.py --old tools/task_verify.py --new tools/task_verify_new.py \\
      --db data/triple_bank.db [--sample 30]
"""
import argparse
import importlib.util
import sqlite3
from collections import Counter


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--sample", type=int, default=30)
    a = ap.parse_args()

    old = load_mod(a.old, "tv_old")
    new = load_mod(a.new, "tv_new")

    c = sqlite3.connect("file:%s?mode=ro" % a.db, uri=True)
    rows = list(c.execute("SELECT id, task, code_text FROM triples ORDER BY id"))

    both_pass = both_fail = newly_fail = newly_pass = 0
    newly = []
    flipped = []
    for i, task, code in rows:
        ro = old.verify(task, code)
        rn = new.verify(task, code)
        po, pn = ro["pass"], rn["pass"]
        if po and pn:
            both_pass += 1
        elif not po and not pn:
            both_fail += 1
        elif po and not pn:
            newly_fail += 1
            bad = [ch["name"] for ch in rn["checks"] if not ch["pass"]]
            newly.append((i, task, bad, rn["quality"]))
        else:
            newly_pass += 1
            flipped.append((i, task))

    # 新拦的归因：按 check 名统计（一条可能多因）
    cnt = Counter()
    for _, _, bad, _ in newly:
        for x in bad:
            cnt[x] += 1

    print("=== 银行规则 A/B ===")
    print("总数 %d | 新旧都过 %d | 新旧都拦 %d | 新拦(增量) %d | 新放行(危险) %d"
          % (len(rows), both_pass, both_fail, newly_fail, newly_pass))
    print("新拦归因:", dict(cnt))
    if flipped:
        print("--- 新放行（规则松动，必须逐条看）---")
        for i, t in flipped[:10]:
            print("  #%d %s" % (i, t[:56]))
    print("--- 新拦抽样 %d 条（判是不是真该拦）---" % min(a.sample, len(newly)))
    for i, t, bad, q in newly[:a.sample]:
        print("  #%-4d q=%.3f %-40s | %s" % (i, q, t[:40], ",".join(bad)))


if __name__ == "__main__":
    main()

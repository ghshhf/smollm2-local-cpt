#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bank_purge_stale.py — 用最新 task_verify 回扫银行，把「要素缺失」的历史垃圾移入 rejects（可回滚）。

2026-10-19：task_verify 补了 22 条规则后回扫，抓出早期裸模套 sin 模板混进银行的样本。
本脚本把这些样本从 triples 移到 rejects（detail 保留原 id + 完整 checks 详情），银行干净且可回滚。

用法:
  python3 bank_purge_stale.py [--dry-run]
  python3 bank_purge_stale.py --db /path/to/triple_bank.db
"""
import sqlite3, json, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DB = os.path.join(os.path.dirname(HERE), "data", "triple_bank.db")


def main():
    dry = "--dry-run" in sys.argv
    db = DB
    for i, a in enumerate(sys.argv):
        if a == "--db" and i + 1 < len(sys.argv):
            db = sys.argv[i + 1]

    import task_verify as tv
    c = sqlite3.connect(db, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=30000")

    rows = list(c.execute(
        "SELECT id, task, code_path, code_text, video_path FROM triples ORDER BY id"))
    moved = []
    for rid, task, cp, code, vp in rows:
        r = tv.verify(task or "", code or "")
        if r["pass"]:
            continue
        detail = json.dumps({"orig_triple_id": rid, "verify": r},
                            ensure_ascii=False)[:4000]
        if not dry:
            c.execute("INSERT INTO rejects(task, code_path, video_path, reason, detail)"
                      " VALUES(?,?,?,?,?)", (task, cp, vp, "task_mismatch", detail))
            c.execute("DELETE FROM triples WHERE id=?", (rid,))
        moved.append((rid, task))

    if not dry:
        c.commit()
    print("=== %s ===" % ("DRY-RUN（未改库）" if dry else "已执行"))
    print("移动/标记 %d 条:" % len(moved))
    for rid, task in moved:
        print("  #%d %s" % (rid, task[:52]))
    print("剩余 triples=%d  rejects=%d" % (
        c.execute("SELECT count(*) FROM triples").fetchone()[0],
        c.execute("SELECT count(*) FROM rejects").fetchone()[0]))


if __name__ == "__main__":
    main()

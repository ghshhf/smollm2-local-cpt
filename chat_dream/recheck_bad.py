#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复核 clean_report.json 里被判坏的那批：用修正后的聚合规则重判，避免误杀。

第一轮用的 min(ss)>3 一票否决太狠，把 25 条中位分 >=6 的好样本也判死了。
"""
import json, sqlite3, sys, time
import chat_dream as cd

DB = "G:/_tmp/data/chat_bank.db"


def main():
    rep = json.load(open("clean_report.json", encoding="utf-8"))
    ids = [b["id"] for b in rep["bad"]]
    cd.load_policy(force=True)
    c = sqlite3.connect(DB)
    rows = c.execute(
        f"select id,q,a,topic from pairs where id in ({','.join('?' * len(ids))}) order by id", ids
    ).fetchall()
    print(f"复核 {len(rows)} 条", flush=True)
    items = [(q, a) for _i, q, a, _t in rows]
    res = {}
    B = 8
    t0 = time.time()
    for s in range(0, len(items), B):
        r = cd.judge_many(items[s:s + B])
        for i in range(len(items[s:s + B])):
            res[s + i] = r.get(i)
        if s % 40 == 0:
            print(f"  {s}/{len(items)} {time.time() - t0:.0f}s", flush=True)

    bad, ok = [], []
    for idx, (pid, q, a, topic) in enumerate(rows):
        v = res.get(idx)
        rec = {"id": pid, "q": q, "a": a, "topic": topic,
               "score": (v or {}).get("score"), "why": (v or {}).get("reason", "")}
        if v is None or not v["pass"]:
            bad.append(rec)
        else:
            ok.append(rec)
    print(f"\n最终判坏 {len(bad)} / 平反 {len(ok)}")
    json.dump({"bad": bad, "rescue": ok}, open("clean_final.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("-> clean_final.json（加 --apply 执行删除）")

    if "--apply" in sys.argv:
        if bad:
            c.executemany("delete from pairs where id=?", [(b["id"],) for b in bad])
            c.commit()
            print(f"已删 {len(bad)} 条")
        print("剩余", c.execute("select count(*) from pairs").fetchone()[0])


if __name__ == "__main__":
    main()

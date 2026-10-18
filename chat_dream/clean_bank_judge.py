#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用新打分机复查全库：把旧打分机（stepfun 单模型、偏松）放进来的垃圾挑出来。

背景：2026-10-19 实测发现，旧 judge 给「主人不要如此，我不会困你。」打 7 分，
而新三模型投票给 2-4 分。银行里已混进一批崩坏样本，必须在下一轮训练前清掉，
否则又是「脏卡污染」重演。

用法:
  python clean_bank_judge.py            # 只扫描，产出 clean_report.json
  python clean_bank_judge.py --apply    # 扫描并删除判定为坏的样本
"""
import json, sqlite3, sys, time
import chat_dream as cd

DB = "G:/_tmp/data/chat_bank.db"


def main():
    apply = "--apply" in sys.argv
    cd.load_policy(force=True)
    c = sqlite3.connect(DB)
    rows = c.execute("select id,q,a,topic from pairs order by id").fetchall()
    print(f"复查 {len(rows)} 条", flush=True)

    items = [(q, a) for _id, q, a, _t in rows]
    verdict = {}
    B = 8
    t0 = time.time()
    for s in range(0, len(items), B):
        chunk = items[s:s + B]
        r = cd.judge_many(chunk)
        for i in range(len(chunk)):
            verdict[s + i] = r.get(i)
        print(f"  {min(s + B, len(items))}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    bad, unsure, good = [], [], []
    for idx, (pid, q, a, topic) in enumerate(rows):
        v = verdict.get(idx)
        if v is None:
            unsure.append({"id": pid, "q": q, "a": a, "topic": topic, "why": "judge失败"})
        elif not v["pass"]:
            bad.append({"id": pid, "q": q, "a": a, "topic": topic,
                        "score": v["score"], "why": v.get("reason", "")})
        else:
            good.append(pid)

    print(f"\n判坏 {len(bad)} / 可疑 {len(unsure)} / 留 {len(good)}")
    json.dump({"bad": bad, "unsure": unsure, "kept": len(good)},
              open("clean_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-> clean_report.json")

    if not apply:
        print("\n（只扫描。确认后加 --apply 执行删除）")
        for b in bad[:15]:
            print(f"  {b['score']:>4} | {b['q'][:18]:<20} | {b['a'][:34]}")
        return

    ids = [b["id"] for b in bad]
    if ids:
        c.executemany("delete from pairs where id=?", [(i,) for i in ids])
        c.commit()
        print(f"已删除 {len(ids)} 条")
    print("剩余", c.execute("select count(*) from pairs").fetchone()[0])


if __name__ == "__main__":
    main()

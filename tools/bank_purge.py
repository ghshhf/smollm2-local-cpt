#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bank_purge.py — 把不满足任务约束的历史样本从 triples 迁到 rejects（可审计、可回滚）

背景（2026-10-18）:
  triple_bank 在装上 task_verify 闸门之前收的 7 条样本全部不合格：
  帧数 7/7 错（要 100/120 帧，一律 frames=40），要素缺失（余弦/方波/三角波/
  李萨如/螺旋/衰减），唯一代码只有 4 个。这些是错误教材，必须移出训练可读区。
  但**不删除**——rejects 是「失败料回灌」的原料，deletes 就断了自进化闭环。

做法:
  1. 逐条跑 task_verify
  2. 不合格 → 插入 rejects(reason='task_mismatch')，并从 triples 删除
  3. 同步从 dream_elite.db 删掉对应桶里 code_text 相同的精英条目
  4. 全部动作写进 data/purge_log_<ts>.json（含原 code_text 的 sha256，可追溯）

用法:
  python tools/bank_purge.py            # dry-run，只报告
  python tools/bank_purge.py --apply    # 真正执行
"""
import os, sys, json, sqlite3, hashlib, time, argparse, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BANK = os.path.join(ROOT, "data", "triple_bank.db")
ELITE = os.path.join(ROOT, "data", "dream_elite.db")


def load_tv():
    spec = importlib.util.spec_from_file_location("task_verify", os.path.join(HERE, "task_verify.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--bank", default=BANK)
    ap.add_argument("--elite", default=ELITE)
    a = ap.parse_args()
    tv = load_tv()

    c = sqlite3.connect(a.bank)
    rows = list(c.execute("select id,task,code_text,code_path,video_path,created_at "
                          "from triples order by id"))
    bad, good = [], []
    for i, task, code, cp, vp, ts in rows:
        r = tv.verify(task or "", code or "")
        rec = {"id": i, "task": task, "code_path": cp, "video_path": vp,
               "created_at": ts, "sha256": hashlib.sha256((code or "").encode()).hexdigest()[:16],
               "pass": r["pass"], "quality": r["quality"],
               "fails": [f"{ch['name']}: {ch['detail']}" for ch in r["checks"] if not ch["pass"]],
               "constraints": r["constraints"]}
        (good if r["pass"] else bad).append(rec)

    print("=" * 76)
    print("triple_bank 审计  共 %d 条  合格 %d  不合格 %d" % (len(rows), len(good), len(bad)))
    print("=" * 76)
    for r in bad:
        print("  ❌ #%s q=%.2f %s" % (r["id"], r["quality"], (r["task"] or "")[:52]))
        for f in r["fails"]:
            print("       - %s" % f)
    for r in good:
        print("  ✅ #%s q=%.2f %s" % (r["id"], r["quality"], (r["task"] or "")[:52]))

    if not a.apply:
        print("\n[dry-run] 加 --apply 执行迁移")
        return
    if not bad:
        print("\n无需迁移")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    logp = os.path.join(ROOT, "data", "purge_log_%s.json" % ts)
    json.dump({"ts": ts, "moved": bad, "kept": good}, open(logp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    for r in bad:
        c.execute("INSERT INTO rejects(task,code_path,video_path,reason,detail) VALUES(?,?,?,?,?)",
                  (r["task"], r["code_path"], r["video_path"], "task_mismatch",
                   json.dumps({"fails": r["fails"], "quality": r["quality"],
                               "purged_from_triples": r["id"], "sha256": r["sha256"]},
                              ensure_ascii=False)[:4000]))
        c.execute("delete from triples where id=?", (r["id"],))
    c.commit()

    # 精英档案同步清理（按 code sha 比对）
    n_elite = 0
    if os.path.isfile(a.elite):
        e = sqlite3.connect(a.elite)
        bad_shas = {}
        for r in bad:
            bad_shas[r["sha256"]] = r["id"]
        for eid, code_text in list(e.execute("select id, code_text from elites")):
            s = hashlib.sha256((code_text or "").encode()).hexdigest()[:16]
            if s in bad_shas:
                e.execute("delete from elites where id=?", (eid,))
                n_elite += 1
        e.commit()
        left = e.execute("select count(*) from elites").fetchone()[0]
        e.close()
        print("精英档案清理 %d 条，剩余 %d 条" % (n_elite, left))

    n_left = c.execute("select count(*) from triples").fetchone()[0]
    n_rej = c.execute("select count(*) from rejects").fetchone()[0]
    print("\n迁移完成：triples %d 条 | rejects %d 条 | 日志 %s" % (n_left, n_rej, logp))
    c.close()


if __name__ == "__main__":
    main()

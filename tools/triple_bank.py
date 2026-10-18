#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""triple_bank.py — 三元组银行（宪法附录B B4/B6 · 实验 E-A）

只收「过了 vid_verify 8 项验收」的 (任务, 代码, 视频) 三元组。
验收是唯一入库闸门 —— 模型不能自己给自己判分（B4）。

用法:
  python tools/triple_bank.py init
  python tools/triple_bank.py add --task "..." --code f.py --video out.mp4 [--ffmpeg PATH]
  python tools/triple_bank.py query --task "..." -k 3 [--show-code]
  python tools/triple_bank.py stats

设计纪律:
  - 拒收也落盘（rejects 表）：失败原因是最有价值的料，做梦态的失败诊断回喂靠它。
  - 检索首版用字符 3-gram Jaccard，不上 embedding —— 先证明"检索有用"再谈精度。
  - 代码内容存快照（不存路径）：文件会被改，库里必须是当时验收通过的那份。
"""
import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "data", "triple_bank.db")
VERIFY = os.path.join(HERE, "vid_verify.py")

SCHEMA = """
CREATE TABLE IF NOT EXISTS triples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task TEXT NOT NULL,
  code_path TEXT,
  code_text TEXT,
  video_path TEXT,
  checks_json TEXT,
  ngrams TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS rejects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task TEXT,
  code_path TEXT,
  video_path TEXT,
  reason TEXT,
  detail TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS trials (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task TEXT NOT NULL,
  used_memory INTEGER NOT NULL,   -- 0=不看库 1=检索top-k后生成
  passed INTEGER NOT NULL,
  bank_size INTEGER,
  note TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def conn():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB)
    c.executescript(SCHEMA)
    return c


def ngrams(s, n=3, cap=400):
    s = re.sub(r"\s+", "", s or "")
    if len(s) < n:
        return []
    g = sorted({s[i:i + n] for i in range(len(s) - n + 1)})
    if len(g) > cap:
        step = len(g) / cap
        g = [g[int(i * step)] for i in range(cap)]
    return g


def jaccard(a, b):
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def verify(video, ffmpeg=None):
    """跑 vid_verify.py --json，返回 (pass, json_dict)。"""
    cmd = [sys.executable, VERIFY, video, "--json"]
    if ffmpeg:
        cmd += ["--ffmpeg", ffmpeg]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "").strip()
    try:
        data = json.loads(out)
    except Exception:
        data = {"pass": False, "raw": out[-800:], "stderr": (p.stderr or "")[-400:]}
    return (p.returncode == 0 and bool(data.get("pass"))), data


def cmd_init(a):
    c = conn()
    n = c.execute("SELECT count(*) FROM triples").fetchone()[0]
    r = c.execute("SELECT count(*) FROM rejects").fetchone()[0]
    print(f"OK 库就绪 {DB}\n    已通过 {n} 条 / 已拒收 {r} 条")


def cmd_add(a):
    if not a.task or not a.code or not a.video:
        print("ERR 三个参数都必填: --task --code --video"); sys.exit(2)
    if not os.path.isfile(a.code):
        print(f"ERR 代码文件不存在: {a.code}"); sys.exit(2)
    c = conn()
    if not os.path.isfile(a.video):
        c.execute("INSERT INTO rejects(task,code_path,video_path,reason,detail) VALUES(?,?,?,?,?)",
                  (a.task, a.code, a.video, "video_missing", "视频文件不存在"))
        c.commit()
        print("REJECT video_missing"); sys.exit(1)

    ok, data = verify(a.video, a.ffmpeg)
    blob = json.dumps(data, ensure_ascii=False)
    if not ok:
        fails = data.get("fails") or data.get("failed") or []
        c.execute("INSERT INTO rejects(task,code_path,video_path,reason,detail) VALUES(?,?,?,?,?)",
                  (a.task, a.code, a.video, "vid_verify_fail", blob[:4000]))
        c.commit()
        print("REJECT vid_verify_fail")
        if isinstance(fails, list) and fails:
            for f in fails[:8]:
                print("   -", f if isinstance(f, str) else json.dumps(f, ensure_ascii=False))
        else:
            print("   -", blob[:300])
        sys.exit(1)

    code_text = open(a.code, encoding="utf-8", errors="replace").read()
    c.execute("""INSERT INTO triples(task,code_path,code_text,video_path,checks_json,ngrams)
                 VALUES(?,?,?,?,?,?)""",
              (a.task, os.path.abspath(a.code), code_text, os.path.abspath(a.video),
               blob, json.dumps(ngrams(a.task), ensure_ascii=False)))
    c.commit()
    n = c.execute("SELECT count(*) FROM triples").fetchone()[0]
    print(f"ACCEPT 入库 #{n}  {a.task[:60]}")

    # P0b/P3: 同时记入 quality-diversity 精英档案（防做梦崩塌 / 经验回放保多样）
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dream_elite", os.path.join(HERE, "dream_elite.py"))
        de = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(de)
        q = float(data.get("quality") or 0.0)
        act, why = de.EliteArchive().add(a.task, code_text,
                                          os.path.abspath(a.video), q,
                                          skill=getattr(a, "skill", None))
        print("ELITE %s (%s)" % (act, why))
    except Exception as e:
        print("ELITE skip (err: %s)" % e)


def cmd_query(a):
    c = conn()
    q = ngrams(a.task)
    rows = c.execute("SELECT id,task,code_path,video_path,ngrams FROM triples").fetchall()
    if not rows:
        print("EMPTY 库是空的"); sys.exit(1)
    scored = []
    for rid, task, cp, vp, ng in rows:
        s = jaccard(q, json.loads(ng or "[]"))
        scored.append((s, rid, task, cp, vp))
    scored.sort(reverse=True)
    for s, rid, task, cp, vp in scored[: a.k]:
        print(f"[{s:.3f}] #{rid}  {task[:70]}")
        print(f"        code={cp}")
        print(f"        video={vp}")
        if a.show_code:
            t = c.execute("SELECT code_text FROM triples WHERE id=?", (rid,)).fetchone()[0]
            print("        ---")
            for line in (t or "").splitlines()[:40]:
                print("        " + line)


def cmd_stats(a):
    c = conn()
    ok = c.execute("SELECT count(*) FROM triples").fetchone()[0]
    rj = c.execute("SELECT reason,count(*) FROM rejects GROUP BY reason").fetchall()
    tr = c.execute("SELECT used_memory,count(*),avg(passed) FROM trials GROUP BY used_memory").fetchall()
    print(f"通过 {ok} 条  库={DB}")
    if rj:
        print("拒收分布:")
        for r, n in rj:
            print(f"  {r}: {n}")
    if tr:
        print("对照实验(首次通过率):")
        for um, n, rate in tr:
            print(f"  memory={'on ' if um else 'off'}  n={n}  pass={(rate or 0) * 100:.0f}%")


def cmd_trial(a):
    """记一条对照结果：不看库 / 看库后生成，是否首次通过。"""
    c = conn()
    bank = c.execute("SELECT count(*) FROM triples").fetchone()[0]
    c.execute("INSERT INTO trials(task,used_memory,passed,bank_size,note) VALUES(?,?,?,?,?)",
              (a.task, 1 if a.memory else 0, 1 if a.passed else 0, bank, a.note))
    c.commit()
    print(f"RECORDED memory={a.memory} passed={a.passed} bank={bank}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(fn=cmd_init)

    p = sub.add_parser("add")
    p.add_argument("--task", required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--skill", default=None,
                   help="技能轴标签(如 三角波/场热力图)，透传给精英档案分桶；不传则正则推断")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("query")
    p.add_argument("--task", required=True)
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--show-code", action="store_true")
    p.set_defaults(fn=cmd_query)

    sub.add_parser("stats").set_defaults(fn=cmd_stats)

    p = sub.add_parser("trial")
    p.add_argument("--task", required=True)
    p.add_argument("--memory", choices=["on", "off"], default="off")
    p.add_argument("--passed", choices=["yes", "no"], default="no")
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_trial)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

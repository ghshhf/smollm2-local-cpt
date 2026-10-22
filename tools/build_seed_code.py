# -*- coding: utf-8 -*-
"""构建代码种子集：合并四批 -> 逐条实跑验证 -> 泄题自查 -> 落盘 jsonl。

用法：
  python tools/build_seed_code.py            # 默认 200 条
  python tools/build_seed_code.py --limit 20 # 只取前 N 条（冒烟用）

产出：
  data/seed_code_200.jsonl      训练用种子（{instruction, output, _type}）
  logs/seed_code_check_200.json 每条的执行输出与泄题自查明细
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import seed_code_p1  # noqa: E402
import seed_code_p2  # noqa: E402
import seed_code_p3  # noqa: E402
import seed_code_p4  # noqa: E402

HOLDOUT = os.path.join(ROOT, "data", "code_holdout_v1.jsonl")
OUT_JSONL = os.path.join(ROOT, "data", "seed_code_200.jsonl")
OUT_LOG = os.path.join(ROOT, "logs", "seed_code_check_200.json")


def norm(s):
    return re.sub(r"\s+", "", s or "")


def ngrams(s, n=3):
    t = norm(s)
    return set(t[i:i + n] for i in range(max(0, len(t) - n + 1)))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_code(code, timeout=20):
    """实跑一条种子代码，返回 (ok, stdout, err)。"""
    path = os.path.join(os.environ.get("TEMP", "."), "_seedchk.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
                           errors="replace")
        return p.returncode == 0, (p.stdout or ""), (p.stderr or "")[:300]
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    seeds = (list(seed_code_p1.S) + list(seed_code_p2.S)
             + list(seed_code_p3.S) + list(seed_code_p4.S))
    if a.limit:
        seeds = seeds[: a.limit]
    print(f"[info] 合并种子 {len(seeds)} 条")

    hold = [json.loads(l) for l in open(HOLDOUT, encoding="utf-8")]
    hold_expect = {norm(t["expect"]) for t in hold if t.get("expect")}
    hold_gram = [(t["id"], ngrams(t["query"])) for t in hold]

    recs, audit, n_run_ok, n_leak = [], [], 0, 0
    t0 = time.time()
    for i, (ins, code, note) in enumerate(seeds, 1):
        ok, out, err = run_code(code)
        if ok and out.strip():
            n_run_ok += 1
        # 泄题自查：① stdout 撞 holdout expect ② 题干 3-gram 相似度过高
        hit_exp = norm(out) in hold_expect
        sims = sorted(((jaccard(ngrams(ins), g), tid) for tid, g in hold_gram), reverse=True)
        top_sim, top_id = sims[0]
        leak = hit_exp or top_sim >= 0.5
        if leak:
            n_leak += 1
        audit.append({
            "i": i, "run_ok": ok, "stdout": out.strip()[:300], "err": err,
            "hit_expect": hit_exp, "top_sim": round(top_sim, 3), "top_id": top_id,
            "leak": leak,
        })
        if not ok:
            print(f"[FAIL-RUN] #{i} {ins[:40]} :: {err[:100]}")
        if leak:
            print(f"[LEAK?]    #{i} sim={top_sim:.2f} vs {top_id} hit_exp={hit_exp} :: {ins[:50]}")

        output = "```python\n" + code + "\n```\n" + note + "\n"
        recs.append({"instruction": ins, "output": output, "_type": "code_rsi"})

    os.makedirs(os.path.dirname(OUT_JSONL) or ".", exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(OUT_LOG) or ".", exist_ok=True)
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        json.dump({"n": len(recs), "run_ok": n_run_ok, "leak": n_leak,
                   "seconds": round(time.time() - t0, 1), "audit": audit},
                  f, ensure_ascii=False, indent=2)

    lens = [len(r["output"]) for r in recs]
    print("=" * 68)
    print(f"[seed] 落盘 {OUT_JSONL}  {len(recs)} 条")
    print(f"  实跑通过(有输出) {n_run_ok}/{len(recs)} = {n_run_ok / len(recs):.0%}")
    print(f"  泄题告警 {n_leak} 条（stdout 撞 expect 或题干相似度 >= 0.5）")
    print(f"  output 字符数: 平均 {sum(lens) / len(lens):.0f}  最小 {min(lens)}  最大 {max(lens)}")
    print(f"  示例:\n---\n{recs[0]['instruction']}\n{recs[0]['output'][:200]}\n---")
    print("=" * 68)


if __name__ == "__main__":
    main()

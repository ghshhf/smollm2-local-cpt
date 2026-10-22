# -*- coding: utf-8 -*-
"""dream_code.py — 代码线做梦器（自进化环的"产粮"那一格）

照 docs/dream-to-module-loop-2026-10.md 的三行定义接：
  做梦 = 行为的发生器（在沙箱里试，用客观闸门筛对的）
本脚本只干"做梦 + 筛"，不做训练；筛出来的进 triple_bank（与自进化同库同表），
再由 export_triples_to_sft.py 导出、sft_full.py 恒维训练，闭环。

与旧做梦器(dream_autorun)的差别：旧的是技能/视频域，闸门是 vid_verify 渲染校验；
代码线没有视频，闸门换成**客观可自动判定的三连**：
  1. 格式闸门：```python 块 + 语言标注 + 块外说明 + <|im_end|> 收尾
  2. 执行闸门：代码真跑得起来（returncode 0 且有 stdout）
  3. 客观闸门：stdout == oracle（oracle 由「种子代码换参数后实跑」自动给出，不需要人判）
  4. 去重闸门：与库内已有代码的 3-gram jaccard < 0.8（防模板复读刷量）

题目从哪来（关键，防泄题/防复述）：
  不拿原题让模型背（那是打分器的活），而是**参数衍生**——把种子题目里的数字换掉，
  同步把种子代码里 print(...) 调用的参数换掉后实跑，得到新题的标准答案(oracle)。
  题目数字变了，抄原代码得不到正确输出，只有真会用公式才过得了闸门。

用法：
  # 小规模冒烟
  python tools/dream_code.py --model models/sft_seed100_03b --max-tasks 6 --k 2
  # 正式产粮
  python tools/dream_code.py --model models/sft_seed100_03b --per-seed 1 --k 3
"""
import argparse
import ast
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

DB = os.path.join(ROOT, "data", "triple_bank.db")
IM_END = "<|im_end|>"
SYSTEM = "你是北极星，主人的私有编程助手。回答给可运行的代码，配一两句关键说明。"
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------- 工具 ----------------
def norm(s):
    return re.sub(r"\s+", "", s or "")


def ngrams(s, n=3, cap=400):
    t = norm(s)
    g = set(t[i:i + n] for i in range(max(0, len(t) - n + 1)))
    return sorted(g)[:cap]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a and b) else 0.0


def run_code(code, timeout=15):
    """执行一段 python，返回 (ok, stdout, stderr)"""
    path = os.path.join(os.environ.get("TEMP", "."), "_dream_code.py")
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


# ---------------- 题目衍生 ----------------
def print_call_nums(code):
    """返回 print(...) 调用里出现的数值字面量的源码位置 [(start, end, value)]。

    只取 print 调用内部的数值 —— 公式常数(如 9.8)在函数体里，不会被误改。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    spans = []
    lines = code.split("\n")
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln) + 1)

    def pos(lineno, col):
        return offs[lineno - 1] + col

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "print":
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)) \
                            and not isinstance(sub.value, bool):
                        s = pos(sub.lineno, sub.col_offset)
                        e = pos(sub.end_lineno, sub.end_col_offset)
                        spans.append((s, e, sub.value))
    spans.sort()
    return spans


def jitter(v, rnd):
    """把参数抖动到邻近但不相同的值：整数仍是整数，量级保持，避免退化题目。"""
    if isinstance(v, int):
        pool = [v + d for d in (-3, -2, -1, 1, 2, 3, 4, 5) if v + d > 0]
        return rnd.choice(pool) if pool else v + 1
    r = rnd.uniform(0.6, 1.8)
    nv = v * r
    return round(nv, 3) if abs(nv) < 1 else round(nv, 2)


def fmt_num(v):
    return str(int(v)) if isinstance(v, int) else ("%g" % v)


def derive(seed, rnd):
    """由一条种子衍生出一道新题 + oracle。失败返回 None。"""
    ins, code = seed["instruction"], seed["code"]
    spans = print_call_nums(code)
    if not spans:
        return None
    ins_nums = NUM_RE.findall(ins)
    # 题目里的数字个数必须和 print 里的数字个数一致，否则无法保证题目与答案对得上
    if len(ins_nums) != len(spans) or len(spans) > 5:
        return None

    new_vals = [jitter(v, rnd) for _, _, v in spans]
    if any(nv == ov for nv, (_, _, ov) in zip(new_vals, spans)):
        return None

    # 1) 改代码：从后往前替换，避免偏移失效
    new_code = code
    for (s, e, _), nv in sorted(zip(spans, new_vals), reverse=True):
        new_code = new_code[:s] + fmt_num(nv) + new_code[e:]
    # 2) 改题目：按出现顺序替换
    it = iter(new_vals)
    new_ins = NUM_RE.sub(lambda m: fmt_num(next(it)), ins)

    ok, out, err = run_code(new_code)
    if not ok or not out.strip():
        return None
    return {"task": new_ins, "oracle": out.strip(), "oracle_code": new_code,
            "seed_ins": ins, "vals": [fmt_num(v) for v in new_vals]}


# ---------------- 闸门 ----------------
def gate_format(text, finished):
    """返回 (ok, code, prose, why)。prose 是代码块外的说明，入库时要一并留着，
    否则 export_triples_to_sft.py 只包代码、不带说明，下一轮就学不会"配说明"。
    """
    # 截断场景下最后一个围栏没闭合，会把半截代码留在"块外说明"里，先切掉
    if text.count("```") % 2 == 1:
        text = text[: text.rfind("```")]
    blocks = re.findall(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", text, re.S)
    if not blocks:
        return False, None, "", "无代码块"
    lang, body = "", None
    for lg, bd in blocks:
        if lg.strip():
            lang, body = lg.strip().lower(), bd
            break
    if body is None:
        lang, body = blocks[0][0], blocks[0][1]
    outside = re.sub(r"```[A-Za-z0-9_+-]*\s*\n.*?```", "", text, flags=re.S)
    outside = outside.replace(IM_END, "").strip()
    if lang != "python":
        return False, body, outside, "语言标注不是 python: %r" % lang
    if len(outside) < 4:
        return False, body, outside, "代码块外没有说明文字"
    # 🔴 不收尾不算失败。2026-10-19 实测：12 个候选里 11 个死在这一条上 ——
    #    模型会把同一个代码块重复生成两三遍直到截断，但它**确实给出了完整的第一块**。
    #    code_eval.py 也只把 has_eos 当统计项、从不作为判据（见 judge_format）。
    #    客观闸门该判的是"代码算得对不对"，不是"会不会停"。停不停是下一轮要练的事。
    if not finished:
        return True, body, outside, "未收尾(截断)，第一块完整"
    return True, body, outside, "ok"


def gate_content(code, oracle, bank_grams):
    """三档验收（2026-10-19 用户要求"打分器别太难"）：

      T2 严格：stdout == oracle          —— 最干净，训练优先用
      T1 近似：oracle 里过半数值出现在输出里 —— 思路对、尾数有偏差
      T0 可跑：执行通过、有非空输出       —— 只保证"是能跑的代码"，结果未核对
      拒绝：语法/运行错、空输出、与库内重复、代码过短

    旧版只有 T2 一档，等于把 0.3B 的产出全毙掉；打分器过严会让环永远空转。
    """
    # 顺序有讲究：先跑、先比对，最后才判长度。
    # 否则 "print(5*10)" 这种答错的短码会被"代码过短"顶掉，看不到真正的诊断。
    ok, out, err = run_code(code)
    if not ok:
        return False, "执行失败: %s" % (err[:120] or "returncode!=0"), -1
    if not out.strip():
        return False, "无 stdout", -1
    if len(code.strip()) < 20:
        return False, "代码过短(疑似取巧)", -1
    g = ngrams(code)
    sim = max((jaccard(g, bg) for bg in bank_grams), default=0.0)
    if sim >= 0.8:
        return False, "与库内代码重复 sim=%.2f" % sim, -1
    if norm(out) == norm(oracle):
        return True, "T2 命中 oracle, 库内最大相似 %.2f" % sim, 2
    want = set(NUM_RE.findall(oracle))
    got = set(NUM_RE.findall(out))
    if want and got:
        hit = len(want & got) / len(want)
        if hit >= 0.5:
            return True, "T1 数值近似 %.0f%%, 库内最大相似 %.2f" % (hit * 100, sim), 1
    if len(norm(out)) >= 4:
        return True, "T0 可运行(结果未核对), 库内最大相似 %.2f" % sim, 0
    return False, "输出过短(%.1f 字符)" % len(norm(out)), -1


# ---------------- 库 ----------------
def bank_conn():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=30000")
    return c


def bank_load(c):
    c.execute("""CREATE TABLE IF NOT EXISTS triples (
        id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT NOT NULL, code_path TEXT,
        code_text TEXT, video_path TEXT, checks_json TEXT, ngrams TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS rejects (
        id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, code_path TEXT,
        video_path TEXT, reason TEXT, detail TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    rows = c.execute("SELECT code_text, ngrams FROM triples").fetchall()
    grams = []
    for ct, ng in rows:
        if ng:
            try:
                grams.append(json.loads(ng))
            except Exception:
                grams.append(ngrams(ct or ""))
        elif ct:
            grams.append(ngrams(ct))
    return grams


def bank_add(c, task, code_text, checks):
    c.execute("""INSERT INTO triples(task,code_path,code_text,video_path,checks_json,ngrams)
                 VALUES(?,?,?,?,?,?)""",
              (task, None, code_text, None,
               json.dumps(checks, ensure_ascii=False), json.dumps(ngrams(code_text), ensure_ascii=False)))
    c.commit()


def bank_reject(c, task, code_text, reason, detail):
    c.execute("""INSERT INTO rejects(task,code_path,video_path,reason,detail)
                 VALUES(?,?,?,?,?)""", (task, None, None, reason, detail[:2000]))
    c.commit()


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default="data/seed_code_200.jsonl")
    ap.add_argument("--llm", default=None, help="tokenizer 路径，默认同 --model")
    ap.add_argument("--per-seed", type=int, default=1, help="每条种子衍生几道题")
    ap.add_argument("--k", type=int, default=3, help="每题采样几个候选")
    ap.add_argument("--max-tasks", type=int, default=0, help="只做前 N 道题(冒烟)")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--threads", default="auto",
                    help="CPU 线程: 数字, 或 auto(默认)=现场标定有效核数")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-save", action="store_true", help="只统计不入库")
    ap.add_argument("--export", default="data/dream_code_sft.jsonl",
                    help="训练就绪 jsonl 落盘路径")
    ap.add_argument("--tag", default="dream")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # 设备/线程一律自适应：换机器、换 GPU 实例都不改参数
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from dev_pick import setup as dev_setup
    device, _n = dev_setup()
    if str(a.threads).strip().lower() not in ("auto", "0", ""):
        torch.set_num_threads(int(a.threads))

    rnd = random.Random(a.seed)
    seeds = []
    for line in open(os.path.join(ROOT, a.seeds), encoding="utf-8"):
        o = json.loads(line)
        m = re.match(r"```python\n(.*?)\n```", o["output"], re.S)
        if m:
            seeds.append({"instruction": o["instruction"], "code": m.group(1)})
    print(f"[seed] 读入 {len(seeds)} 条种子")

    # 衍生题目
    tasks = []
    for s in seeds:
        for _ in range(a.per_seed):
            d = derive(s, rnd)
            if d:
                tasks.append(d)
    # 同一道题可能被多次命中，去重
    seen, uniq = set(), []
    for t in tasks:
        if t["task"] in seen:
            continue
        seen.add(t["task"])
        uniq.append(t)
    tasks = uniq
    rnd.shuffle(tasks)
    if a.max_tasks:
        tasks = tasks[: a.max_tasks]
    print(f"[task] 可参数化衍生 {len(tasks)} 道新题（每道采样 {a.k} 个候选）")

    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, a.llm or a.model))
    # 🔴 不要传 dtype=torch.float32：新版 transformers 会把它塞进 config.torch_dtype，
    #    紧接着打印 config 时 json.dumps(dtype 对象) 直接崩
    #    （TypeError: Object of type dtype is not JSON serializable）。
    #    权重本身就是 fp32（1.39GB = 346.9M × 4B），直接加载即可。
    m = AutoModelForCausalLM.from_pretrained(os.path.join(ROOT, a.model))
    if next(m.parameters()).dtype != torch.float32:
        m = m.float()
    m.eval()
    m = m.to(device)
    eos = tok.convert_tokens_to_ids(IM_END)
    c = bank_conn()
    grams = bank_load(c)
    print(f"[bank] 库内已有 {len(grams)} 条")

    t0 = time.time()
    n_accept = n_reject = 0
    stats = {"no_code": 0, "exec_fail": 0, "mismatch": 0, "dup": 0, "fmt": 0,
             "t0": 0, "t1": 0, "t2": 0}
    log = []
    for i, t in enumerate(tasks, 1):
        prompt = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
                  "<|im_start|>assistant\n" % (SYSTEM, t["task"]))
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        best = None
        with torch.no_grad():
            for _ in range(a.k):
                out = m.generate(ids, max_new_tokens=a.max_new, do_sample=True,
                                 temperature=a.temp, top_p=a.top_p,
                                 eos_token_id=eos, pad_token_id=tok.pad_token_id or eos)
                gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
                finished = IM_END in gen
                gen = gen.split(IM_END)[0]
                fmt_ok, code, prose, why = gate_format(gen, finished)
                if not fmt_ok:
                    stats["fmt"] += 1
                    if best is None:
                        best = (code, False, "格式: " + why, prose, -1)
                    continue
                ok, why2, tier = gate_content(code, t["oracle"], grams)
                if ok:
                    best = (code, True, why2, prose, tier)
                    break
                if best is None or not best[1]:
                    best = (code, False, why2, prose, -1)
                key = "dup" if "重复" in why2 else ("exec_fail" if "执行" in why2 else "mismatch")
                stats[key] += 1
        if best is None:
            stats["no_code"] += 1
            n_reject += 1
            print(f"[{i:3d}/{len(tasks)}] REJECT 无产出  {t['task'][:44]}")
            continue
        code, passed, why, prose, tier = best
        if passed:
            n_accept += 1
            stats["t%d" % tier] += 1
            if not a.no_save:
                bank_add(c, t["task"], code, {"oracle": t["oracle"], "why": why,
                                              "vals": t["vals"], "seed": t["seed_ins"],
                                              "prose": prose, "tier": tier})
                grams.append(ngrams(code))
            print(f"[{i:3d}/{len(tasks)}] ACCEPT[T{tier}] {t['task'][:38]} | {why[:40]}")
            log.append({"task": t["task"], "code": code, "prose": prose,
                        "oracle": t["oracle"], "tier": tier})
        else:
            n_reject += 1
            if not a.no_save and code:
                bank_reject(c, t["task"], code, "code_gate", why)
            print(f"[{i:3d}/{len(tasks)}] REJECT {why[:52]} | {t['task'][:34]}")

    el = time.time() - t0
    print("=" * 66)
    print(f"[{a.tag}] 做梦 {len(tasks)} 题 × {a.k} 候选, 用时 {el / 60:.1f} 分")
    print(f"  入库 {n_accept}  拒绝 {n_reject}  "
          f"(格式 {stats['fmt']} / 执行 {stats['exec_fail']} / 不符 {stats['mismatch']} / 重复 {stats['dup']})")
    print(f"  分档 T2 {stats['t2']} / T1 {stats['t1']} / T0 {stats['t0']}")
    print(f"  接受率 {n_accept / max(1, len(tasks)):.0%}")
    print("=" * 66)
    if not a.no_save and log:
        p = os.path.join(ROOT, "logs", f"dream_code_{a.tag}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in log:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[save] 本轮产物明细 {p}")
        # 训练就绪格式：与 data/seed_code_200.jsonl 完全同构，可直接喂 sft_full.py
        ep = os.path.join(ROOT, a.export)
        with open(ep, "w", encoding="utf-8") as f:
            for r in log:
                out = "```python\n" + r["code"].strip("\n") + "\n```\n" + (r["prose"] or "").strip() + "\n"
                f.write(json.dumps({"instruction": r["task"], "output": out,
                                    "_type": "code_dream", "tier": r["tier"]},
                                   ensure_ascii=False) + "\n")
        print(f"[save] 训练就绪 {ep} ({len(log)} 条) —— 下一轮可直接 --data {a.export}")


if __name__ == "__main__":
    main()

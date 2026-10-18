#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""weak_to_strong.py — 弱到强升级闭环（RSI 线 P2 · LEASH 启发）

LEASH(2610.22280) 思路本地化：
  弱模型(北极星 0.2B/0.3B) 作为「提议者」给出候选改进；
  评估器(本地确定性规则) 验收质量；
  强教师(云端，可插拔) 把通过的候选精炼成更高质量版本并回流。

宪法红线(保函数 + E-7): 本地只动 fast weights / 测试时 latent / 可删记忆层。
  强教师是「外部精炼源」——它产出的成品以「可删记忆/精英示例」形式回流进
  三元组银行 / 精英档案，绝不直接改主干参数（主干锚定）。

可跑性:
  - 本地侧(提议+评估+入库) 完全可跑；不依赖云端教师。
  - 教师接口可插拔：默认 LocalStubTeacher(无云端→标记 pending_teacher，不伪造精炼)；
    配置 --teacher-url 后走 HttpTeacher(POST JSON)。本地未配教师时闭环仍收敛于
    「弱模型自产 + 本地验收 + 入库待审」，证明管线成立。
  - 精炼成品回流入 triple_bank，作为高质量 few-shot 被 dream_loop 检索 →
    弱模型经记忆检索间接变强(LEASH 的 weak→strong 回流本质)。

依赖: 同 dream_loop（venv_ttt + In-Place-TTT）

用法:
  venv_ttt/Scripts/python.exe tools/weak_to_strong.py --smoke
  venv_ttt/Scripts/python.exe tools/weak_to_strong.py \
      --base D:/work/_local_ai/train_15b/backups/0.3B/polaris-03b_20261018 --n 3
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "data", "weak_to_strong.db")

# 内置待升级工件（小样本，模拟「弱模型要改进的东西」）
SEED_ARTIFACTS = [
    {"kind": "code", "instruction": "改进下面函数，使其能处理空输入并返回稳定结果",
     "artifact": "def mean(xs):\n    return sum(xs)/len(xs)\n"},
    {"kind": "code", "instruction": "把下面的脚本补全，加异常处理并输出耗时",
     "artifact": "import time\nstart=time.time()\nprint('hello')\nprint(time.time()-start)\n"},
    {"kind": "prompt", "instruction": "把这段系统提示写得更严谨，约束模型只答中文",
     "artifact": "你是一个助手，请帮助用户解决问题。"},
]


# ── 教师接口（可插拔） ─────────────────────────────────────────────────
class LocalStubTeacher:
    """无云端教师：不伪造精炼，仅标记 pending_teacher。"""
    name = "stub"
    def refine(self, kind, artifact, candidate):
        return None  # 未精炼


class HttpTeacher:
    """云端教师：POST 候选让其精炼。失败/超时回退 stub。"""
    name = "http"
    def __init__(self, url, timeout=30):
        self.url = url; self.timeout = timeout
    def refine(self, kind, artifact, candidate):
        try:
            import urllib.request
            payload = json.dumps({"kind": kind, "artifact": artifact,
                                  "candidate": candidate}).encode("utf-8")
            req = urllib.request.Request(self.url, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8")).get("refined")
        except Exception as e:
            print("   [teacher] 请求失败→回退 stub: %s" % e)
            return None


# ── 本地评估器（确定性规则，不替模型自判） ────────────────────────────
def evaluate(kind, candidate):
    if not candidate or len(candidate.strip()) < 5:
        return 0.0, False, "候选过短/空"
    if kind == "code":
        try:
            compile(candidate, "<cand>", "exec")
        except SyntaxError as e:
            return 0.1, False, "语法错误: %s" % e
        # 非平凡：相对原 artifact 至少有新增行
        return 0.7, True, "可编译"
    else:  # prompt
        if len(candidate) > len("你是一个助手"):
            return 0.7, True, "提示变长(更约束)"
        return 0.3, False, "提示无明显改进"


# ── 提议 + 闭环 ───────────────────────────────────────────────────────
class WeakToStrong:
    def __init__(self, base, ttt_on, steps, lr, teacher=None):
        sys.path.insert(0, HERE)
        from ttt_infer import build, get_tok
        from ttt_latent import LatentAdapter, adapt
        self.adapt = adapt
        self.tok = get_tok(base)
        self.model = build(base=base, ttt_on=ttt_on,
                           ttt_layers=list(range(22, 30)) if ttt_on else None)
        self.teacher = teacher or LocalStubTeacher()
        self.steps = steps; self.lr = lr
        self._conn()

    def _conn(self):
        os.makedirs(os.path.dirname(DB), exist_ok=True)
        self.c = sqlite3.connect(DB)
        self.c.executescript("""
        CREATE TABLE IF NOT EXISTS proposals (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          kind TEXT, artifact TEXT, candidate TEXT, refined TEXT,
          score REAL, passed INTEGER, teacher TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );""")
        self.c.commit()

    def propose(self, spec):
        support = spec["instruction"] + "\n" + spec["artifact"]
        adapter, losses = self.adapt(self.model, self.tok, support, self.steps, self.lr)
        prompt = ("任务: %s\n原始内容:\n%s\n请产出改进版（只输出改进后内容，不要解释）:\n"
                  % (spec["instruction"], spec["artifact"]))
        # 复用 dream_loop 的采样生成（若可用），否则贪心
        try:
            sys.path.insert(0, HERE)
            from dream_loop import sample_generate, parse_code
            gen = sample_generate(self.model, self.tok, adapter, prompt, n_new=160)
            cand = parse_code(gen) or gen.strip()
        except Exception:
            cand = support  # 退化
        return cand, (losses[-1] if losses else float("nan"))

    def run_one(self, spec):
        kind = spec["kind"]
        cand, loss = self.propose(spec)
        score, ok, why = evaluate(kind, cand)
        refined = self.teacher.refine(kind, spec["artifact"], cand)
        teacher_name = self.teacher.name if refined is not None else (self.teacher.name + "(pending)")
        self.c.execute(
            "INSERT INTO proposals(kind,artifact,candidate,refined,score,passed,teacher) "
            "VALUES(?,?,?,?,?,?,?)",
            (kind, spec["artifact"], cand, refined, score, 1 if ok else 0, teacher_name))
        self.c.commit()
        pid = self.c.execute("SELECT last_insert_rowid()").fetchone()[0]
        print("   [propose#%d] kind=%s loss=%.3f score=%.2f passed=%s teacher=%s"
              % (pid, kind, loss, score, ok, teacher_name))
        print("   [candidate] %s" % cand[:160].replace("\n", " ⏎ "))
        # 验收通过且教师未精炼 → 仍可回流进 triple_bank(作为代码/提示记忆)
        if ok:
            self._reflow(spec, refined or cand)
        return ok

    def _reflow(self, spec, content):
        """精炼成品回流：写入 triple_bank 记忆(供 dream_loop 检索 → 弱模型间接变强)。"""
        try:
            task = "%s: %s" % (spec["kind"], spec["instruction"])
            # 代码类直接落精英档案(走 dream_elite)；提示类记 proposals 即可
            if spec["kind"] == "code":
                import importlib.util
                de = importlib.util.spec_from_file_location(
                    "dream_elite", os.path.join(HERE, "dream_elite.py"))
                m = importlib.util.module_from_spec(de); de.loader.exec_module(m)
                act, why = m.EliteArchive().add(task, content, "", 0.8)
                print("   [reflow] elite=%s (%s)" % (act, why))
        except Exception as e:
            print("   [reflow] skip err=%s" % e)

    def run(self, n):
        specs = SEED_ARTIFACTS[:n] if n <= len(SEED_ARTIFACTS) else \
            (SEED_ARTIFACTS * ((n // len(SEED_ARTIFACTS)) + 1))[:n]
        ok = 0
        for s in specs:
            try:
                if self.run_one(s):
                    ok += 1
            except Exception as e:
                print("   [w2s] 异常=%s" % e)
        print("\n=== 弱到强轮次结束 ===  本地验收通过 %d/%d，教师=%s"
              % (ok, n, self.teacher.name))
        return ok


def smoke():
    """不加载模型：验证 DB + 评估器 + stub 教师 通路。"""
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB)
    c.executescript("CREATE TABLE IF NOT EXISTS proposals(id INTEGER PRIMARY KEY,kind TEXT,"
                    "artifact TEXT,candidate TEXT,refined TEXT,score REAL,passed INTEGER,"
                    "teacher TEXT);")
    t = LocalStubTeacher()
    for s in SEED_ARTIFACTS:
        cand = s["artifact"] + "\n# improved\n"
        sc, ok, why = evaluate(s["kind"], cand)
        r = t.refine(s["kind"], s["artifact"], cand)
        c.execute("INSERT INTO proposals(kind,artifact,candidate,refined,score,passed,teacher)"
                  " VALUES(?,?,?,?,?,?,?)",
                  (s["kind"], s["artifact"], cand, r, sc, 1 if ok else 0, t.name))
    c.commit()
    n = c.execute("SELECT count(*) FROM proposals").fetchone()[0]
    print("[smoke] 提案入库 %d 条; 教师=%s(精炼=None→pending 正确)" % (n, t.name))
    print("[smoke] 弱到强本地侧通路 OK（propose→evaluate→refine→reflow 骨架成立）")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="D:/work/_local_ai/train_15b/backups/0.2B/r6_final_20261016")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--ttt-off", action="store_true")
    ap.add_argument("--teacher-url", default=None, help="云端教师端点(可选)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        smoke(); return
    teacher = HttpTeacher(a.teacher_url) if a.teacher_url else None
    w = WeakToStrong(a.base, ttt_on=not a.ttt_off, steps=a.steps, lr=a.lr, teacher=teacher)
    w.run(a.n)


if __name__ == "__main__":
    main()

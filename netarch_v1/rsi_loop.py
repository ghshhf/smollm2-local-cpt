#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSI 闭环 V1 骨架 —— 三权分立 + 确定性预言机裁判。

设计依据（docs/netarch/v1_04_verdict_rsi_evaluator_20261014.md，gemini+nemotron 强趋同裁决）：
  - 留出法不够（同源评估器与执行器共享归纳偏置会共谋）⇒ 执行器/评估器/审计器必须【异源+三权分立】；
  - 最终胜负不交 LLM 自评，交【确定性预言机】（AST 检查/pytest/REPL/时延探针）；
  - 「自知是模型」= 选择性预测损失（可预测 [I_DONT_KNOW]，不确定性区不敢放水）。

骨架第一版只跑通「提议→执行→确定性裁决→(失败沉淀负样本 | 成功原子提交)」这条最小闭环，
不接真模型（用 stub 打桩），目的是把状态机/裁判/回滚/审计的骨架先立住，真模型后接。

红线：RSI 第一版只允许改沙箱里的规则/代码，主线只读；任何自修改都要有版本+回滚+审计。

跑法：
  python rsi_loop.py demo        # 跑一轮完整闭环 demo（含一次失败、一次成功）
  python rsi_loop.py states      # 看状态机定义
"""
import sys, json, time, uuid, hashlib, subprocess, tempfile, os, sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "rsi.db")

# ── 三权分立角色（异源；真实部署时执行器=0.5B三值训练态，评估器=0.4B fp16）────────
ROLES = {
    "executor":  "M_pi   (0.5B BitNet 三值, 训练态) —— 负责提出解/写代码",
    "critic":    "M_crit (0.4B fp16, 异源, 温度0) —— 负责挑毛病/打分, 但不作最终裁决",
    "oracle":    "Deterministic Oracle (AST/pytest/REPL) —— 最终胜负【唯一】裁决者",
    "auditor":   "M_audit —— 独立留痕(版本/权限/回滚路径), 不参与决策",
}

# ── 状态机定义（V1-04 裁决 §二）──────────────────────────────────────────────
STATES = [
    "S0_COGNITIVE_HOLE_SCAN",   # Δ-NLL/高困惑触发
    "S1_PROBLEM_CONSTRUCTION",  # 构造 输入-预期-断言
    "S2_PROVE",                 # 执行器求解
    "S2_FALSIFY",               # 反例器注入对抗边界
    "S3_ORACLE_JUDGE",          # 确定性裁决
    "S4_REJECT_NEG_SAMPLE",     # 失败→沉淀负样本(死信队列)
    "S5_GAIN_EVAL",             # 成功→测Golden集有无退化
    "S6_ROLLBACK",              # 负迁移→硬回滚
    "S7_ATOMIC_COMMIT",         # 无退化且收益>0→原子提交
]

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def db():
    c = sqlite3.connect(DB, timeout=30); c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA busy_timeout=30000")
    return c

def init_db():
    c = db(); c.executescript("""
    CREATE TABLE IF NOT EXISTS rsi_runs(
      run_id TEXT PRIMARY KEY, state TEXT, role TEXT, verdict TEXT,
      payload TEXT, created_at TEXT, updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS oracle_tasks(       -- 试题：输入-断言-执行器
      task_id TEXT PRIMARY KEY, run_id TEXT, input TEXT, assertion TEXT,
      candidate TEXT, oracle_result TEXT, passed INTEGER, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS neg_samples(       -- S4 死信队列(不可变)
      id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, reason TEXT, sample TEXT, at TEXT
    );
    CREATE TABLE IF NOT EXISTS commits(           -- S7 原子提交(带回滚指针)
      commit_id TEXT PRIMARY KEY, run_id TEXT, version_before TEXT, version_after TEXT,
      gain REAL, rollback_of TEXT, at TEXT
    );
    CREATE TABLE IF NOT EXISTS audit_log(         -- M_audit 独立留痕
      id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT, action TEXT, detail TEXT, at TEXT
    );
    """); c.commit(); c.close()

def audit(actor, action, detail=""):
    c = db(); c.execute("INSERT INTO audit_log(actor,action,detail,at) VALUES(?,?,?,?)",
                        (actor, action, detail, now())); c.commit(); c.close()

def set_state(run_id, state, role="executor", verdict="", payload=""):
    c = db()
    c.execute("INSERT OR REPLACE INTO rsi_runs(run_id,state,role,verdict,payload,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
              (run_id, state, role, verdict, payload, now(), now()))
    c.commit(); c.close()
    audit("loop", f"state->{state}", verdict)

# ── 确定性预言机（唯一裁决者）：真的跑 Python/pytest，不用 LLM 自评 ──────────
def deterministic_oracle(candidate_code: str, test_code: str, timeout=15):
    """在临时目录写 candidate+test，用 python 跑起来。返回 (passed, stdout_tail)。
    这是'非对称信息博弈'的落地点：模型无法自评，只有真实执行说了算。"""
    with tempfile.TemporaryDirectory() as td:
        cand = os.path.join(td, "cand.py"); test = os.path.join(td, "test_cand.py")
        open(cand, "w", encoding="utf-8").write(candidate_code)
        open(test, "w", encoding="utf-8").write(test_code)
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", test],
                               capture_output=True, text=True, timeout=timeout, cwd=td)
            return (r.returncode == 0), (r.stdout or r.stderr)[-400:]
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

# ── stub 模型（第一版打桩；真模型后接）────────────────────────────────────────
def stub_executor(task_input):
    # 故意对某些任务给错解，用来演示 S4 负样本；真模型此处替换
    return "def solve(x):\n    return x * 2\n"

def stub_propose_problem():
    """S1 构造：输入-预期-断言。"""
    return {
        "input": "solve(3)",
        "assertion": "assert solve(3) == 6",
        "test": "from cand import solve\ndef test_solve():\n    assert solve(3) == 6\n",
    }

def check_negative_transfer(run_id):
    """S5：测 Golden 集有无退化。骨架里用版本比对打桩。"""
    return {"golden_delta": 0.0, "regressed": False}

# ── 主闭环 ─────────────────────────────────────────────────────────────────
def one_round(run_id):
    # S0 认知空洞扫描（骨架：桩触发）
    set_state(run_id, "S0_COGNITIVE_HOLE_SCAN", payload="stub:Δ-NLL trigger")
    # S1 构造试题
    set_state(run_id, "S1_PROBLEM_CONSTRUCTION")
    prob = stub_propose_problem()
    task_id = "task_" + uuid.uuid4().hex[:8]
    c = db(); c.execute("INSERT INTO oracle_tasks(task_id,run_id,input,assertion,created_at) VALUES(?,?,?,?,?)",
                        (task_id, run_id, prob["input"], prob["assertion"], now())); c.commit(); c.close()
    # S2a 执行器求解
    set_state(run_id, "S2_PROVE", role="executor")
    candidate = stub_executor(prob["input"])
    c = db(); c.execute("UPDATE oracle_tasks SET candidate=? WHERE task_id=?", (candidate, task_id)); c.commit(); c.close()
    # S3 确定性裁决（唯一裁决者）
    set_state(run_id, "S3_ORACLE_JUDGE", role="oracle")
    passed, out = deterministic_oracle(candidate, prob["test"])
    c = db(); c.execute("UPDATE oracle_tasks SET oracle_result=?, passed=? WHERE task_id=?",
                        (out, 1 if passed else 0, task_id)); c.commit(); c.close()
    if not passed:
        # S4 失败→沉淀负样本（死信队列，不变）
        set_state(run_id, "S4_REJECT_NEG_SAMPLE", verdict="FAIL", payload=out)
        c = db(); c.execute("INSERT INTO neg_samples(run_id,reason,sample,at) VALUES(?,?,?,?)",
                            (run_id, "oracle_fail", candidate, now())); c.commit(); c.close()
        audit("oracle", "reject", "candidate failed deterministic test")
        return "REJECTED"
    # S5 收益评估（Golden 集退化检测）
    set_state(run_id, "S5_GAIN_EVAL")
    gt = check_negative_transfer(run_id)
    if gt["regressed"]:
        set_state(run_id, "S6_ROLLBACK", verdict="NEG_TRANSFER", payload=json.dumps(gt))
        audit("auditor", "rollback", "negative transfer detected")
        return "ROLLED_BACK"
    # S7 原子提交
    commit_id = "cmt_" + uuid.uuid4().hex[:8]
    c = db(); c.execute("INSERT INTO commits(commit_id,run_id,version_before,version_after,gain,rollback_of,at) VALUES(?,?,?,?,?,?,?)",
                        (commit_id, run_id, "v_prev", "v_next", gt["golden_delta"], None, now())); c.commit(); c.close()
    set_state(run_id, "S7_ATOMIC_COMMIT", verdict="PASS", payload=commit_id)
    audit("auditor", "commit", f"atomic commit {commit_id}")
    return "COMMITTED"

def demo():
    init_db()
    print("="*66); print("RSI 闭环 V1 骨架 demo —— 三权分立 + 确定性预言机"); print("="*66)
    for k, v in ROLES.items():
        print(f"  [{k:9s}] {v}")
    print(f"\n状态机（{len(STATES)} 态）: " + " → ".join(s.split('_')[0] for s in STATES))
    # 跑两轮：一轮成功、一轮制造失败
    print("\n--- round 1 (期望 COMMITTED，candidate 恰好满足断言) ---")
    r1 = one_round("run_" + uuid.uuid4().hex[:8])
    print("  结果:", r1)
    print("--- round 2 (制造 oracle 失败 → 应进 S4 死信队列) ---")
    # 注入一个错的 candidate
    global stub_executor
    orig = stub_executor
    stub_executor = lambda ti: "def solve(x):\n    return x * 999\n"   # 故意错
    r2 = one_round("run_" + uuid.uuid4().hex[:8])
    stub_executor = orig
    print("  结果:", r2)
    # 汇总
    c = db()
    print("\n--- 审计/落库 ---")
    print("  runs:", c.execute("SELECT COUNT(*) FROM rsi_runs").fetchone()[0])
    print("  neg_samples(死信):", c.execute("SELECT COUNT(*) FROM neg_samples").fetchone()[0])
    print("  commits:", c.execute("SELECT COUNT(*) FROM commits").fetchone()[0])
    print("  audit_log:", c.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0])
    c.close()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo": demo()
    elif cmd == "states":
        for s in STATES: print(s)
    else: print(__doc__)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task_verify.py — 任务约束符合度校验（做梦闭环的第 7 道闸门）

🔴 为什么必须有这一层（2026-10-18 实测发现）:
    triple_bank 里 7 条「验签通过」的三元组，只有 4 个唯一代码，两两 3-gram
    相似度 0.84~1.00，而且没有一条实现自己任务的要求：
      #3 要 100 帧，代码 frames=40
      #4 要 0~6π / 120 帧 / 振幅衰减，代码 4π / frames=40 / 振幅恒定
      #6 要「正弦+余弦叠加」，代码只有 np.sin(x)
      #9 要李萨如曲线、#10 要螺旋线，代码还是 sin 波相位平移
    根因：vid_verify 的 6 项全是「视频能不能播 / 画面动不动」，属于**产物层**验收；
    它永远不知道代码做的是不是题目要求的事。质检缺失 ⇒ 做梦产出的「精英」是错误教材。

设计原则（刻意克制）:
  - 纯静态，不执行任何生成代码（执行级校验属 code_verify 的另一层）
  - 只判**能高置信机验**的约束；抽不到或判不了的项标 skipped，不计入分母，
    避免用脆规则误杀（"精 = 先读懂数据再定规则"）
  - 与 vid_verify 同构：checks 列表 + score + hard + quality + pass，
    便于 dream_loop 直接串在 vid_verify 之后

可判约束（从任务文本抽取）:
  1. frames   「N 帧」                 vs 代码 frames=N
  2. domain   「0~Kπ」                vs 代码 linspace 上界 == K*pi
  3. outfile  「保存为 X.mp4」         vs 代码 save('X')
  4. symbols  关键词→必需符号(余弦/方波/三角波/衰减/散点/柱状/饼/热力/向量场...)
  5. anim     任务含「动画」           vs 代码有 FuncAnimation/ArtistAnimation 且 save

用法:
  python tools/task_verify.py --task "..." --code-file path.py [--json]
  python tools/task_verify.py --bank [--db data/triple_bank.db] [--json]
退出码: 0=PASS  1=FAIL
"""
import os, sys, re, json, argparse, sqlite3, ast

# 关键词 → 代码里必须出现的 token（任一命中即算）
SYMBOL_RULES = [
    ("余弦",      ["np.cos", "cos(", "numpy.cos"]),
    ("正切",      ["np.tan", "tan("]),
    ("方波",      ["square(", "sign(", "sgn(", "np.where", "heaviside"]),
    ("三角波",    ["sawtooth(", "triangle", "arcsin", "abs(", "tri("]),
    ("衰减",      ["exp(", "np.exp", "** -", "**(-", "1 / (1", "1/(1"]),
    ("散点",      ["scatter("]),
    ("柱状",      ["bar(", "barh("]),
    ("条形",      ["bar(", "barh("]),
    ("饼图",      ["pie("]),
    ("热力图",    ["imshow(", "heatmap", "pcolormesh"]),
    ("等高线",    ["contour(", "contourf("]),
    ("向量场",    ["quiver("]),
    ("直方图",    ["hist("]),
    ("螺旋",      ["cos(2", "cos(3", "2 * np.pi * t", "2*np.pi*t", "helix", "spiral"]),
    ("李萨如",    ["lissajous", "sin(3", "sin(2 * t)", "sin(5"]),
    ("三维",      ["Axes3D", "projection='3d'", 'projection="3d"']),
    ("柱体",      ["bar3d("]),
    ("曲面",      ["plot_surface("]),
    ("粒子",      ["scatter(", "Line2D", "plot("]),
    ("文本",      ["text(", "annotate("]),
    ("网格",      ["grid("]),
    ("图例",      ["legend("]),
]


# 「要素缺失」的软硬分层（2026-10-18 自检发现：一律软会让「缺余弦」蒙混过关）
#   HARD: 有唯一标准写法的函数，缺了就是真没做（np.cos / np.sign(sin) / arcsin / exp）
#   SOFT: 写法很不唯一（螺旋可以极坐标也可以参数方程），硬判会误杀
HARD_SYMBOLS = {"余弦", "正切", "方波", "三角波", "衰减", "散点", "柱状", "饼图",
                "热力图", "等高线", "向量场", "直方图", "三维", "柱体", "曲面"}
SOFT_SYMBOLS = {"螺旋", "李萨如", "粒子", "文本", "网格", "图例"}


def extract_constraints(task):
    """从任务文本抽约束。返回 dict；抽不到的键不存在。"""
    c = {}
    t = task or ""

    m = re.search(r"(\d+)\s*帧", t)
    if m:
        c["frames"] = int(m.group(1))

    m = re.search(r"0\s*[~～至到\-]\s*(\d+(?:\.\d+)?)\s*[πp]i?", t)
    if m:
        c["domain_pi"] = float(m.group(1))

    m = re.search(r"保存为\s*([\w./\\-]+\.(?:mp4|gif|avi|webm))", t)
    if m:
        c["outfile"] = m.group(1)

    m = re.search(r"(\d+(?:\.\d+)?)\s*秒", t)
    if m:
        c["seconds"] = float(m.group(1))

    m = re.search(r"(\d+)\s*fps", t, re.I)
    if m:
        c["fps"] = int(m.group(1))

    if "动画" in t or "动图" in t:
        c["animated"] = True

    syms = [kw for kw, _ in SYMBOL_RULES if kw in t]
    if syms:
        c["symbols"] = syms
    return c


def _num(node):
    """从 AST 节点静态求值成 float（失败返回 None）。支持常数/加减乘除/np.pi。"""
    try:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in ("pi", "PI"):
            return None  # 裸 pi 无法确定数值，交给上层判
        if isinstance(node, ast.Attribute) and node.attr in ("pi", "PI"):
            return 3.141592653589793
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = _num(node.operand)
            return -v if v is not None else None
        if isinstance(node, ast.BinOp):
            a, b = _num(node.left), _num(node.right)
            if a is None or b is None:
                return None
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Div) and b != 0:
                return a / b
            if isinstance(node.op, ast.Pow):
                return a ** b
    except Exception:
        return None
    return None


def _is_pi_expr(node):
    """判断表达式是否含 π（用于 linspace 上界识别）。"""
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in ("pi", "PI"):
            return True
        if isinstance(n, ast.Name) and n.id in ("pi", "PI"):
            return True
    return False


def analyze_code(code):
    """静态抽取代码事实。不执行。"""
    facts = {"frames": None, "linspace_uppers": [], "linspace_has_pi": False,
             "save_name": None, "has_anim": False, "fps": None,
             "tokens": set(), "ok_parse": False, "error": None}
    try:
        tree = ast.parse(code or "")
        facts["ok_parse"] = True
    except SyntaxError as e:
        facts["error"] = "SyntaxError: %s" % e
        tree = None

    # token 集合（文本级，供 symbol 规则用）
    low = (code or "").lower().replace(" ", "")
    facts["tokens"] = low

    if tree is None:
        return facts

    def call_name(node):
        f = node.func
        if isinstance(f, ast.Attribute):
            return (getattr(f.value, "id", "") + "." + f.attr).lower(), f.attr.lower()
        if isinstance(f, ast.Name):
            return f.id.lower(), f.id.lower()
        return "", ""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted, base = call_name(node)

        # frames / fps：关键字参数或位置参数
        for kw in node.keywords:
            if kw.arg == "frames":
                v = _num(kw.value)
                if v is not None:
                    facts["frames"] = int(v)
                else:
                    # 可能是 len(x) / 变量 -> 试图找赋值
                    facts["frames_expr"] = ast.unparse(kw.value) if hasattr(ast, "unparse") else "?"
            if kw.arg == "fps":
                v = _num(kw.value)
                if v is not None:
                    facts["fps"] = int(v)

        if base in ("funcanimation", "artistanimation"):
            facts["has_anim"] = True

        if base == "linspace" and len(node.args) >= 2:
            upper = node.args[1]
            facts["linspace_has_pi"] = facts["linspace_has_pi"] or _is_pi_expr(upper)
            v = _num(upper)
            if v is not None:
                # 收全部上界：3D 曲面/热力图会建 X/Y 两个网格，只留最后一个会漏判
                facts["linspace_uppers"].append(v)
            elif hasattr(ast, "unparse"):
                facts.setdefault("linspace_exprs", []).append(ast.unparse(upper))

        if base == "save" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                facts["save_name"] = a0.value

    # 补充：frames= 也可能是普通赋值后传入，扫一遍 Assign
    if facts["frames"] is None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
               and isinstance(node.value.value, int):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and "frame" in tgt.id.lower():
                        facts["frames"] = facts["frames"] or None  # 不采信，交由调用点
    return facts


def verify(task, code):
    """返回 {checks, quality, pass, constraints}。"""
    cons = extract_constraints(task)
    facts = analyze_code(code)
    checks = []

    def chk(name, ok, detail, hard=True, score=None):
        checks.append({"name": name, "pass": bool(ok), "detail": detail,
                       "score": (1.0 if ok else 0.0) if score is None else score,
                       "hard": bool(hard)})

    if not facts["ok_parse"]:
        chk("代码可解析", False, facts["error"] or "parse failed", hard=True)
        return {"checks": checks, "quality": 0.0, "pass": False,
                "constraints": cons, "facts_summary": {"ok_parse": False}}

    # 1. frames
    if "frames" in cons:
        want = cons["frames"]
        got = facts["frames"]
        if got is None:
            chk("帧数符合", False,
                "任务要求 %d 帧，代码里未找到常量 frames=N（可能为变量/表达式）" % want)
        else:
            chk("帧数符合", got == want, "任务 %d 帧 / 代码 %d 帧" % (want, got))

    # 2. domain（任一 linspace 上界命中即可：3D 曲面会建 X/Y 两个同域网格）
    if "domain_pi" in cons:
        want = cons["domain_pi"] * 3.141592653589793
        uppers = facts["linspace_uppers"]
        if not uppers:
            expr = ", ".join(facts.get("linspace_exprs", [])) or "未找到 linspace"
            chk("定义域符合", False, "任务 0~%gπ，代码上界表达式=%s" % (cons["domain_pi"], expr))
        else:
            ok = any(abs(u - want) < 1e-6 for u in uppers)
            chk("定义域符合", ok, "任务 0~%gπ (%.4f) / 代码上界 %s"
                % (cons["domain_pi"], want, ["%.4f" % u for u in uppers]))

    # 3. outfile
    if "outfile" in cons:
        want = os.path.basename(cons["outfile"])
        got = facts["save_name"]
        ok = bool(got) and os.path.basename(got) == want
        chk("输出文件名符合", ok, "任务 %s / 代码 %s" % (want, got))

    # 4. symbols（分层：缺标准函数→硬拦；缺写法不唯一的→软告警）
    #    取舍：做梦产物重跑一次很便宜（几秒），错误教材入库的污染代价极高
    #    （会被 few-shot 检索反复放大，见 check_pollution）。故宁可严。
    if cons.get("symbols"):
        toks = facts["tokens"]
        missing = []
        for kw in cons["symbols"]:
            pats = dict(SYMBOL_RULES)[kw]
            if not any(p.lower().replace(" ", "") in toks for p in pats):
                missing.append(kw)
        hard_missing = [m for m in missing if m in HARD_SYMBOLS]
        chk("任务要素齐备", not missing,
            "缺: %s" % "、".join(missing) if missing else "全中: %s" % "、".join(cons["symbols"]),
            hard=bool(hard_missing))

    # 5. animated
    if cons.get("animated"):
        ok = facts["has_anim"] and bool(facts["save_name"])
        chk("产出动画", ok,
            "FuncAnimation=%s save=%s" % (facts["has_anim"], facts["save_name"]))

    if not checks:
        return {"checks": checks, "quality": 1.0, "pass": True,
                "constraints": cons, "facts_summary": {"note": "无可机验约束，跳过"}}

    quality = sum(c["score"] for c in checks) / len(checks)
    hard_fail = any(c["hard"] and not c["pass"] for c in checks)
    return {"checks": checks, "quality": round(quality, 4), "pass": not hard_fail,
            "constraints": cons,
            "facts_summary": {k: v for k, v in facts.items() if k != "tokens"}}


def selftest():
    """正例 + 反例自检。只有反例测得出「漏检」，测不出「误杀」——
    误杀比漏检更贵（会把合格产物挡在银行外，做梦白跑），所以正例必须一起测。"""
    GOOD = [
        ("用 matplotlib 画一个衰减正弦波动画(0~6π, 120 帧，振幅随时间衰减)，保存为 out.mp4",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\nimport numpy as np\n\n"
         "fig, ax = plt.subplots()\nx = np.linspace(0, 6 * np.pi, 200)\n"
         "line, = ax.plot(x, np.exp(-0.05 * x) * np.sin(x))\n\ndef upd(i):\n"
         "    line.set_ydata(np.exp(-0.05 * x) * np.sin(x + i / 10.0))\n    return line,\n\n"
         "ani = FuncAnimation(fig, upd, frames=120, blit=False)\n"
         "ani.save('out.mp4', writer='ffmpeg', fps=10)\n"),
        ("用 matplotlib 画正弦+余弦叠加动画(0~4π, 100 帧)，保存为 out.mp4",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\nimport numpy as np\n\n"
         "fig, ax = plt.subplots()\nx = np.linspace(0, 4 * np.pi, 300)\n"
         "l1, = ax.plot(x, np.sin(x))\nl2, = ax.plot(x, np.cos(x))\n\ndef upd(i):\n"
         "    l2.set_ydata(np.cos(x + i / 9.0))\n    return l1, l2\n\n"
         "ani = FuncAnimation(fig, upd, frames=100)\nani.save('out.mp4', writer='ffmpeg', fps=10)\n"),
        ("随便画一张有意思的图",   # 无可机验约束 → 应放行（不能因为抽不出约束就卡死）
         "import matplotlib\nmatplotlib.use('Agg')\nprint('hi')\n"),
    ]
    BAD = [
        ("用 matplotlib 画一个正弦波动画(0~4π, 100 帧)，保存为 out.mp4",
         "import numpy as np\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\n"
         "x=np.linspace(0,4*np.pi,200)\n"
         "ani=FuncAnimation(plt.gcf(), lambda i: None, frames=40)\n"
         "ani.save('out.mp4')\n", "帧数"),
        ("用 matplotlib 画正弦+余弦叠加的动画(0~4π, 100 帧)，保存为 out.mp4",
         "import numpy as np\nfrom matplotlib.animation import FuncAnimation\n"
         "x=np.linspace(0,4*np.pi,200)\n"
         "ani=FuncAnimation(None, lambda i: None, frames=100)\nani.save('out.mp4')\n", "要素"),
    ]
    n_ok = 0
    for task, code in GOOD:
        r = verify(task, code)
        ok = r["pass"]
        n_ok += 1 if ok else 0
        print("  [%s] 正例  %s" % ("PASS" if ok else "FAIL", task[:44]))
        if not ok:
            for ch in r["checks"]:
                if not ch["pass"]:
                    print("         ! %s: %s" % (ch["name"], ch["detail"]))
    for task, code, expect_catch in BAD:
        r = verify(task, code)
        ok = not r["pass"]
        n_ok += 1 if ok else 0
        print("  [%s] 反例应拦住(%s)  %s" % ("PASS" if ok else "FAIL", expect_catch, task[:34]))
    total = len(GOOD) + len(BAD)
    print("\n自检 %d/%d" % (n_ok, total))
    return n_ok == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--task")
    ap.add_argument("--code-file")
    ap.add_argument("--bank", action="store_true", help="回扫 triple_bank 全表")
    ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "triple_bank.db"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    if a.bank:
        c = sqlite3.connect(a.db)
        rows = list(c.execute("select id, task, code_text from triples order by id"))
        results, npass = [], 0
        for i, task, code in rows:
            r = verify(task, code)
            r["id"] = i
            r["task"] = task
            results.append(r)
            npass += 1 if r["pass"] else 0
        if a.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print("=" * 78)
            print("triple_bank 任务符合度回扫  共 %d 条  通过 %d  失败 %d"
                  % (len(rows), npass, len(rows) - npass))
            print("=" * 78)
            for r in results:
                print("\n#%s %s" % (r["id"], "PASS ✅" if r["pass"] else "FAIL ❌"))
                print("  task: %s" % r["task"])
                for ch in r["checks"]:
                    mark = "PASS" if ch["pass"] else ("FAIL" if ch["hard"] else "WARN")
                    print("    [%-4s] %-12s %s" % (mark, ch["name"], ch["detail"]))
        sys.exit(0 if npass == len(rows) else 1)

    if not a.code_file:
        ap.error("需要 --code-file 或 --bank")
    code = open(a.code_file, encoding="utf-8", errors="replace").read()
    task = a.task or ""
    r = verify(task, code)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print("=" * 70)
        print("任务:", task)
        print("=" * 70)
        print("约束:", json.dumps(r["constraints"], ensure_ascii=False))
        for ch in r["checks"]:
            mark = "PASS" if ch["pass"] else ("FAIL" if ch["hard"] else "WARN")
            print("  [%-4s] %-12s %s" % (mark, ch["name"], ch["detail"]))
        print("  结论: %s   quality=%.2f" % ("PASS ✅" if r["pass"] else "FAIL ❌", r["quality"]))
    sys.exit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()

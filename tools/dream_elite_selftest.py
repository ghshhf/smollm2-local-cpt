#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dream_elite_selftest.py — 精英档案自检（改桶规则/闸门后必须跑）

三组检查，全绿才算没把 RSI 的多样化设施改坏：
  A. 三重闸门：近重复 / 桶容量 / 新桶直通（临时库，绝不碰真档案）
  B. 桶划分分离度：从三元组银行回算，看桶数是否退化（退化=又开始塌桶）
  C. 技能标签一致性：make_spec 模板显式 skill 与 infer_skill 正则是否一致

用法:
  PYTHONIOENCODING=utf-8 venv/Scripts/python.exe tools/dream_elite_selftest.py
"""
import os
import sqlite3
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dream_elite as de   # noqa: E402

FAKE_CODE = {
    "plot": "import matplotlib.pyplot as plt\nimport numpy as np\nx=np.linspace(0,1,50)\n"
            "ax=plt.subplots()[1]\nax.plot(np.sin(x))\n"
            "from matplotlib.animation import FuncAnimation\n",
    "plot2": "import matplotlib.pyplot as plt\nimport numpy as np\nt=np.linspace(0,6,80)\n"
             "fig,ax=plt.subplots()\nax.plot(np.cos(t)*np.exp(-0.1*t))\n"
             "from matplotlib.animation import FuncAnimation\n",
    "surface": "import matplotlib.pyplot as plt\nimport numpy as np\nz=np.arange(0,10,0.1)\n"
               "fig=plt.figure()\nax=fig.add_subplot(111,projection='3d')\n"
               "ax.plot_surface(z,z,z)\nfrom matplotlib.animation import FuncAnimation\n",
    "scatter": "import matplotlib.pyplot as plt\nimport numpy as np\nq=np.random.randn(100)\n"
               "fig,ax=plt.subplots()\nax.scatter(q,q)\n"
               "from matplotlib.animation import FuncAnimation\n",
}


def check_gates():
    tmp = os.path.join(tempfile.mkdtemp(), "t.db")
    c = sqlite3.connect(tmp)
    c.executescript("""
    CREATE TABLE elites (id INTEGER PRIMARY KEY AUTOINCREMENT, bucket TEXT NOT NULL,
     domain TEXT NOT NULL, sig TEXT NOT NULL, quality REAL NOT NULL, task TEXT,
     code_text TEXT, video_path TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    """)
    ar = de.EliteArchive(c)
    A, B, C, D = (FAKE_CODE[k] for k in ("plot", "plot2", "surface", "scatter"))
    cases = [
        ("新桶直通",     "画正弦波动画", A, 0.8, "三角波", "keep"),
        ("同码同分",     "画正弦波动画", A, 0.8, "三角波", "skip"),
        ("同码更高分",   "画正弦波动画", A, 0.9, "三角波", "keep"),
        ("桶内第2条",    "画衰减正弦",   B, 0.8, "三角波", "keep"),
        ("桶内第3条(散点)", "画余弦波",  D, 0.8, "三角波", "keep"),
        ("桶满且最差",   "画正弦波v2",   D + "\n# v2", 0.7, "三角波", "skip"),
        ("新技能低分",   "画 3D 曲面",   C, 0.5, "曲面三维", "keep"),
        ("兜底正则推断", "画李萨如曲线", D.replace("scatter", "plot"), 0.8, None, "keep"),
    ]
    bad = 0
    print("── A. 三重闸门（临时库）")
    for name, task, code, q, sk, want in cases:
        got, _ = ar.add(task, code, "/tmp/v.mp4", q, skill=sk)
        ok = (got == want)
        bad += (not ok)
        print("  %s %-16s want=%-4s got=%-4s" % ("OK " if ok else "!!!", name, want, got))
    return bad, len(cases)


def check_spread():
    """从银行回算桶数：桶数 <= 2 说明又塌桶了（旧规则就是 2）。"""
    bank = os.path.join(ROOT, "data", "triple_bank.db")
    if not os.path.exists(bank):
        print("── B. 桶分离度：银行不存在，跳过")
        return 0, 0
    c = sqlite3.connect(bank)
    rows = c.execute("SELECT task,code_text FROM triples").fetchall()
    buckets = set(de.bucket_of(t, code) for t, code in rows)
    print("── B. 桶分离度：银行 %d 条 → %d 个桶" % (len(rows), len(buckets)))
    bad = 1 if (len(rows) >= 4 and len(buckets) <= 2) else 0
    if bad:
        print("  !!! 塌桶：%d 条只落进 %d 个桶" % (len(rows), len(buckets)))
    return bad, 1


def check_skill_labels():
    """模板显式 skill 与正则推断是否一致（不一致=兜底规则有偏差）。"""
    try:
        import dream_autorun as da
    except Exception as e:
        print("── C. 技能标签：autorun 导入失败(%s)，跳过" % e)
        return 0, 0
    bad = 0
    print("── C. 技能标签一致性")
    for skill, tpl, _hint in da.TEMPLATES:
        task = tpl.format(A=3, F=100)
        inf = de.infer_skill(task)
        ok = (inf == skill)
        bad += (not ok)
        if not ok:
            print("  !!! 显式=%-8s 推断=%-8s %s" % (skill, inf, task[:44]))
    print("  %d/%d 一致，技能轴覆盖 %d 类" %
          (len(da.TEMPLATES) - bad, len(da.TEMPLATES),
           len(set(s for s, _, _ in da.TEMPLATES))))
    return bad, len(da.TEMPLATES)


def main():
    print("=== dream_elite 自检 ===")
    b1, n1 = check_gates()
    b2, n2 = check_spread()
    b3, n3 = check_skill_labels()
    bad, n = b1 + b2 + b3, n1 + n2 + n3
    print("\n结果：%d/%d 通过%s" % (n - bad, n, "" if bad == 0 else "  ← 有失败项"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dream_elite.py — 做梦 quality-diversity 精英档案（EliteReplay 启发，P0b/P3）

问题: 做梦自产任务若只用二元 vid_verify 验收，会行为锐化 —— 模型反复产出同一类解，
       bank 变成无聊列表，fast weights 只被单一模式巩固 → 退化。
解决: 维护一个 MAP-Elites 式精英档案，按「任务域 × 代码风格」网格保多样高质量代表。
       - 每个行为桶(bucket)只留 top-K 精英（按 quality 排序），且强制不同代码签名
       - 新候选要么质量够高、要么落入未见过的桶/签名 → 才入库，否则丢弃
       - 这把「保住 base 的多种解法」(EliteReplay 结论) 落到本地做梦闭环

用法:
  python tools/dream_elite.py add --task "..." --code f.py --video out.mp4 --quality 0.9
  python tools/dream_elite.py sample -k 5
  python tools/dream_elite.py stats
"""
import argparse, json, os, re, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "data", "dream_elite.db")
K_PER_BUCKET = 3  # 每个行为桶保留的精英数

# 任务域关键词（与本地六能力/做梦任务相关；命中第一个即归属）
TASK_DOMAINS = [
    ("排序搜索", r"排序|搜索|查找|二分|哈希|队列|栈|树|图|遍历"),
    ("文件IO", r"文件|读写|csv|json|解析|目录|批量命名|重命名"),
    ("网络爬虫", r"爬虫|请求|http|下载|接口|api|抓取|代理"),
    ("数学计算", r"数学|计算|方程|矩阵|统计|素数|阶乘|积分"),
    ("文本处理", r"文本|字符串|正则|分词|替换|摘要|翻译|编码"),
    ("游戏模拟", r"游戏|模拟|贪吃蛇|2048|生命游戏|棋|随机"),
    ("UI可视化", r"界面|可视化|图表|绘图|gui|tkinter|pygame|动画"),
    ("算法结构", r"递归|动态规划|回溯|分治|并查集|堆|链表"),
]
# 代码风格签名（用于桶内多样性判别）
CODE_SIGS = [
    (r"\bdef\b", "func"), (r"\bclass\b", "class"), (r"\bfor\b|\bwhile\b", "loop"),
    (r"递归|recurs|\bself\.call|\breturn \w+\(", "recur"), (r"re\.", "regex"),
    (r"requests\.|urllib|http", "net"), (r"open\(|csv\.|json\.load", "io"),
    (r"np\.|numpy|torch\.|matplotlib", "scipy"), (r"tkinter|pygame|pyqt", "gui"),
    (r"threading|asyncio|multiprocess", "conc"),
]


def task_domain(task):
    t = (task or "")
    for name, pat in TASK_DOMAINS:
        if re.search(pat, t, re.I):
            return name
    return "其他"


def code_sigs(code):
    c = code or ""
    return tuple(sorted(s for pat, s in CODE_SIGS if re.search(pat, c)))


def bucket_of(task, code):
    return "%s|%s" % (task_domain(task), "+".join(code_sigs(code)) or "bare")


def conn():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS elites (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      bucket TEXT NOT NULL,
      domain TEXT NOT NULL,
      sig TEXT NOT NULL,
      quality REAL NOT NULL,
      task TEXT,
      code_text TEXT,
      video_path TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_bucket ON elites(bucket);
    """)
    return c


class EliteArchive:
    def __init__(self, c=None):
        self.c = c or conn()

    def add(self, task, code_text, video, quality):
        quality = float(quality)
        bucket = bucket_of(task, code_text)
        domain = task_domain(task)
        sig = "+".join(code_sigs(code_text)) or "bare"
        # 桶内已存在同签名精英？则只在质量更高时替换
        same_sig = self.c.execute(
            "SELECT id,quality FROM elites WHERE bucket=? AND sig=?", (bucket, sig)).fetchall()
        if same_sig:
            best = max(q for _, q in same_sig)
            if quality <= best:
                return ("skip", "同桶同签名已有质量更高精英 q=%.2f" % best)
            old = max(same_sig, key=lambda r: r[1])[0]
            self.c.execute("DELETE FROM elites WHERE id=?", (old,))
        # 桶容量控制：超额且本候选非桶内最高 → 丢弃（保多样+保优）
        in_bucket = self.c.execute(
            "SELECT id,quality FROM elites WHERE bucket=?", (bucket,)).fetchall()
        if len(in_bucket) >= K_PER_BUCKET and quality <= max(q for _, q in in_bucket):
            return ("skip", "桶满且非最高 q=%.2f" % max(q for _, q in in_bucket))
        self.c.execute(
            "INSERT INTO elites(bucket,domain,sig,quality,task,code_text,video_path) "
            "VALUES(?,?,?,?,?,?,?)",
            (bucket, domain, sig, quality, task, code_text, video))
        # 桶超额后清掉最低
        if len(in_bucket) + 1 > K_PER_BUCKET:
            worst = self.c.execute(
                "SELECT id FROM elites WHERE bucket=? ORDER BY quality ASC LIMIT 1",
                (bucket,)).fetchone()[0]
            self.c.execute("DELETE FROM elites WHERE id=?", (worst,))
        self.c.commit()
        return ("keep", "bucket=%s q=%.2f" % (bucket, quality))

    def sample(self, k=5):
        rows = self.c.execute(
            "SELECT bucket,domain,sig,quality,task FROM elites "
            "ORDER BY quality DESC").fetchall()
        return rows[:k]

    def stats(self):
        total = self.c.execute("SELECT count(*) FROM elites").fetchone()[0]
        buckets = self.c.execute(
            "SELECT domain,count(*),avg(quality) FROM elites GROUP BY domain").fetchall()
        return total, buckets


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("--task", required=True); p.add_argument("--code", required=True)
    p.add_argument("--video", default=""); p.add_argument("--quality", type=float, required=True)
    p.set_defaults(fn="add")
    p = sub.add_parser("sample"); p.add_argument("-k", type=int, default=5); p.set_defaults(fn="sample")
    p = sub.add_parser("stats"); p.set_defaults(fn="stats")
    a = ap.parse_args()

    arch = EliteArchive()
    if a.fn == "add":
        code_text = open(a.code, encoding="utf-8", errors="replace").read()
        act, why = arch.add(a.task, code_text, a.video, a.quality)
        print("%s %s" % (act.upper(), why))
    elif a.fn == "sample":
        for b, dom, sig, q, task in arch.sample(a.k):
            print("[%.2f] %s | %s | %s" % (q, dom, sig, (task or "")[:50]))
    elif a.fn == "stats":
        total, buckets = arch.stats()
        print("精英总数 %d" % total)
        for dom, n, q in sorted(buckets, key=lambda r: -r[1]):
            print("  %s: %d 个, 均质 %.2f" % (dom, n, q or 0))


if __name__ == "__main__":
    main()

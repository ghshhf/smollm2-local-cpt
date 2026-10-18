#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dream_elite.py — 做梦 quality-diversity 精英档案（EliteReplay 启发，P0b/P3）

问题: 做梦自产任务若只用二元 vid_verify 验收，会行为锐化 —— 模型反复产出同一类解，
       bank 变成无聊列表，fast weights 只被单一模式巩固 → 退化。
解决: 维护一个 MAP-Elites 式精英档案，按「技能轴 × 技术轴」二维网格保多样高质量代表。

桶划分（2026-10-18 重构）:
  旧版 bucket = 任务域关键词 | 代码布尔签名 → 实测 9 条银行产物塌进 2 个桶
  （8 条全落到 "UI可视化|func+scipy"，因为域正则只认到 "动画"、签名只认到 matplotlib）。
  新版拆成两轴:
    技能轴 skill  = 这个做梦任务在练什么能力（三角波/衰减振荡/参数曲线/分形/场热力…）
                    —— 生成端知道答案就显式传 --skill，不知道才退回正则推断。
    技术轴 tech   = 代码怎么画出来的（viz_kind 画法 + anim_driver 驱动方式）
  这两轴正交，才真正区分"同一能力不同实现"与"不同能力同样实现"。

准入规则（三重闸门）:
  1. 代码近重复闸门: 与同桶已有精英代码 3-gram Jaccard > DUP_JACCARD 判为重复解，
     只有 quality 更高才替换 —— 防止 sin 图画 10 遍把档案撑满。
  2. 桶容量闸门: 每桶保 top-K，超额且非桶内最高 → 丢弃（保多样+保优）。
  3. 新桶直通: 落入未见过的桶 → 直接收（这是新技能，质量再低也有研究价值）。

用法:
  python tools/dream_elite.py add --task "..." --code f.py --video out.mp4 --quality 0.9 [--skill 三角波]
  python tools/dream_elite.py sample -k 5
  python tools/dream_elite.py stats
  python tools/dream_elite.py backfill          # 从三元组银行全量重建（幂等，改桶规则后必跑）
"""
import argparse, json, os, re, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "data", "dream_elite.db")
BANK = os.path.join(ROOT, "data", "triple_bank.db")
K_PER_BUCKET = 3   # 每个行为桶保留的精英数
DUP_JACCARD = 0.82  # 代码 3-gram 相似度超过此值 → 视为同一解法的重复

sys.path.insert(0, HERE)
from triple_bank import ngrams, jaccard  # noqa: E402  复用银行现成的 3-gram 工具

# ── 技能轴：练的是什么能力（先匹配先赢，specific → general）─────────────
SKILL_HINTS = [
    ("分形",       r"julia|mandelbrot|分形|逃逸时间"),
    ("曲面三维",   r"曲面|三维|3d|view_init|projection\s*=\s*['\"]3d"),
    ("场热力图",   r"热力图|热力|波动方程|pde|温度场|势场"),
    ("级数逼近",   r"傅里叶|fourier|级数|逼近|泰勒"),
    ("粒子统计",   r"布朗|粒子|随机游走|散点|直方图|分布"),
    # 排序类必须排在「排名条形图」之前：任务同时含"排序"和"柱状"时，技能轴认排序
    # （"哪根柱子高"只是技术呈现，技术轴 viz_kind 已单独记 条形+funcanim）。
    ("排序搜索",   r"排序|冒泡|插入排序|快速排序|搜索|查找|二分|哈希|队列|栈|遍历"),
    ("排名条形图", r"条形|柱状|排名|bar\s*race|竞赛"),
    ("参数曲线",   r"李萨如|lissajous|参数曲线|摆线|玫瑰线"),
    ("极坐标曲线", r"螺旋|极坐标|polar"),
    ("衰减振荡",   r"衰减|阻尼|衰减正弦"),
    ("多序列对比", r"叠加|对比|多曲线|多条|方波|三角波|锯齿"),
    ("三角波",     r"正弦|余弦|\bsin\b|\bcos\b|波形"),
    ("刚体动画",   r"弹跳|小球|碰撞|重力|抛体|钟摆"),
    ("生命游戏",   r"生命游戏|game of life|元胞|康威"),
    ("文件IO",     r"文件|读写|csv|解析|目录|批量命名|重命名"),
    ("网络请求",   r"爬虫|请求|http|下载|接口|api|抓取"),
    ("数学计算",   r"计算|方程|矩阵|统计|素数|阶乘|积分"),
    ("文本处理",   r"文本|字符串|正则|分词|替换|摘要|翻译"),
    ("算法结构",   r"递归|动态规划|回溯|分治|并查集|链表"),
]
# 视觉类任务的兜底标记（技能轴识别不出但确实是画图）
VIZ_FALLBACK = "可视化其他"

# ── 技术轴之一：画法（specific → general）──────────────────────────────
VIZ_KINDS = [
    ("曲面",   r"plot_surface|plot_wireframe|projection\s*=\s*['\"]3d|Axes3D"),
    ("图像",   r"imshow|pcolormesh|contourf|set_array|matshow"),
    ("散点",   r"\.scatter\("),
    ("条形",   r"\.barh?\(|set_width|set_height"),
    ("填充",   r"fill_between|\.fill\("),
    ("饼图",   r"\.pie\("),
    ("极坐标", r"polar|subplot\(.*polar"),
    ("折线",   r"\.plot\("),
    ("像素",   r"cv2\.|PIL|Image\.new|tobytes"),
]
# ── 技术轴之二：动画驱动方式 ───────────────────────────────────────────
ANIM_DRIVERS = [
    ("funcanim",   r"FuncAnimation"),
    ("artistanim", r"ArtistAnimation"),
    ("writer",     r"VideoWriter|imageio|mimsave|av\.open|ffmpeg.*pipe"),
    ("多帧存图",   r"savefig\(.*%0?\d*d|savefig\(.*\{.*\}"),
]

# ── 代码风格签名（仅供人看/调试，不再参与桶键）─────────────────────────
CODE_SIGS = [
    (r"\bdef\b", "func"), (r"\bclass\b", "class"), (r"\bfor\b|\bwhile\b", "loop"),
    (r"递归|recurs|\bself\.call", "recur"), (r"re\.", "regex"),
    (r"requests\.|urllib", "net"), (r"open\(|csv\.|json\.load", "io"),
    (r"matplotlib", "mpl"), (r"np\.|numpy", "np"), (r"torch\.", "torch"),
    (r"cv2\.", "cv2"), (r"FuncAnimation|ArtistAnimation", "anim"),
    (r"set_data|set_ydata|set_array|set_width|set_height|set_offsets", "stateupd"),
    (r"np\.(cumsum|linspace|arange|meshgrid|fft|abs)", "vec"),
    (r"threading|asyncio|multiprocess", "conc"),
]


def infer_skill(task):
    """从任务文本推断技能轴；识别不出且明显是画图 → 可视化其他。"""
    t = task or ""
    for name, pat in SKILL_HINTS:
        if re.search(pat, t, re.I):
            return name
    if re.search(r"matplotlib|画|动画|图表|绘图|可视化|mp4", t, re.I):
        return VIZ_FALLBACK
    return "其他"


def viz_kind(code):
    c = code or ""
    for name, pat in VIZ_KINDS:
        if re.search(pat, c, re.I):
            return name
    return "无绘图"


def anim_driver(code):
    c = code or ""
    for name, pat in ANIM_DRIVERS:
        if re.search(pat, c, re.I):
            return name
    return "静态"


def code_sigs(code):
    c = code or ""
    return tuple(sorted(s for pat, s in CODE_SIGS if re.search(pat, c)))


def tech_of(code):
    return "%s+%s" % (viz_kind(code), anim_driver(code))


def bucket_of(task, code, skill=None):
    """桶键 = 技能轴 | 技术轴。旧版是「任务域|全签名」，实测会塌成一两个桶。"""
    return "%s|%s" % (skill or infer_skill(task), tech_of(code))


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

    def _dup_hit(self, bucket, code_text):
        """同桶内已存在近重复解法？返回 (是否重复, 该精英质量)。"""
        g = ngrams(code_text or "")
        best_q, hit = -1.0, False
        for q, old in self.c.execute(
                "SELECT quality,code_text FROM elites WHERE bucket=?", (bucket,)):
            if jaccard(g, ngrams(old or "")) >= DUP_JACCARD:
                hit = True
                best_q = max(best_q, q)
        return hit, best_q

    def add(self, task, code_text, video, quality, skill=None):
        quality = float(quality)
        skill = skill or infer_skill(task)
        bucket = bucket_of(task, code_text, skill)
        tech = tech_of(code_text)
        sig = "+".join(code_sigs(code_text)) or "bare"

        # 闸门 1：代码近重复 —— 同一解法重画一遍不该占坑
        dup, dup_q = self._dup_hit(bucket, code_text)
        if dup and quality <= dup_q:
            return ("skip", "近重复解法(3gram>=%.2f)且质量不更高 q=%.2f" % (DUP_JACCARD, dup_q))
        if dup:
            # 质量更高：清掉近重复的旧条目再收
            g = ngrams(code_text or "")
            for i, old in self.c.execute(
                    "SELECT id,code_text FROM elites WHERE bucket=?", (bucket,)).fetchall():
                if jaccard(g, ngrams(old or "")) >= DUP_JACCARD:
                    self.c.execute("DELETE FROM elites WHERE id=?", (i,))

        # 闸门 2：桶容量
        in_bucket = self.c.execute(
            "SELECT id,quality FROM elites WHERE bucket=?", (bucket,)).fetchall()

        # 闸门 3：全新桶直通（新能力，质量低也有研究价值）
        if len(in_bucket) >= K_PER_BUCKET and quality <= min(q for _, q in in_bucket):
            return ("skip", "桶满(%d)且质量不占优 q=%.2f" % (len(in_bucket), quality))

        self.c.execute(
            "INSERT INTO elites(bucket,domain,sig,quality,task,code_text,video_path) "
            "VALUES(?,?,?,?,?,?,?)",
            (bucket, skill, tech, quality, task, code_text, video))
        if len(in_bucket) + 1 > K_PER_BUCKET:
            worst = self.c.execute(
                "SELECT id FROM elites WHERE bucket=? ORDER BY quality ASC LIMIT 1",
                (bucket,)).fetchone()[0]
            self.c.execute("DELETE FROM elites WHERE id=?", (worst,))
        self.c.commit()
        return ("keep", "bucket=%s q=%.2f" % (bucket, quality))

    def sample(self, k=5):
        return self.c.execute(
            "SELECT bucket,domain,sig,quality,task FROM elites "
            "ORDER BY quality DESC").fetchall()[:k]

    def diverse(self, k=1, avoid_skills=None, min_len=120):
        """按「桶多样性」取 k 条，每桶只取质量最高那条。

        与 sample() 的分工：sample 是「取最好的」，本方法是「取不一样的」。
        MAP-Elites 的价值全在回放多样性——few-shot 里塞一条别的技能的示例，
        能压住「所有任务都往同一个模板上套」的塌缩（实测发生过：15 个任务
        全生成 matplotlib 动画，银行 9 条产物只落 2 个桶）。

        avoid_skills: 已在 prompt 里出现过的技能名，跳过它们免得重复喂同一路。
        返回 [{bucket,skill,quality,task,code}]，code 已截断到 1600 字。

        🔴 min_len：低于此长度的代码不进 few-shot。实测坑：冒烟测试的
        `print('DONE')`（35 字）也占了一个桶，且质量并列时按 id 排在最前，
        会被优先选中喂给模型 —— 一条什么都教不了的示例比没有示例更糟。
        """
        avoid = set(avoid_skills or [])
        rows = self.c.execute(
            "SELECT bucket,domain,quality,task,code_text FROM elites "
            "ORDER BY quality DESC").fetchall()
        seen, out = set(), []
        for bucket, domain, q, task, code in rows:
            if bucket in seen or domain in avoid:
                continue
            code = (code or "").strip()
            if len(code) < min_len:
                continue
            seen.add(bucket)
            out.append({"bucket": bucket, "skill": domain, "quality": q,
                        "task": task or "", "code": code[:1600]})
            if len(out) >= k:
                break
        return out

    def stats(self):
        total = self.c.execute("SELECT count(*) FROM elites").fetchone()[0]
        buckets = self.c.execute(
            "SELECT domain,count(*),avg(quality) FROM elites GROUP BY domain "
            "ORDER BY count(*) DESC").fetchall()
        nb = self.c.execute("SELECT count(DISTINCT bucket) FROM elites").fetchone()[0]
        return total, nb, buckets

    def rebuild_from_bank(self):
        """从三元组银行全量重建（改桶规则后必跑）。幂等：先清空再逐条 add。"""
        if not os.path.exists(BANK):
            return 0, 0
        b = sqlite3.connect(BANK)
        rows = b.execute(
            "SELECT task,code_text,video_path,checks_json FROM triples ORDER BY id").fetchall()
        self.c.execute("DELETE FROM elites")
        self.c.commit()
        kept = 0
        for task, code, video, cj in rows:
            try:
                q = float(json.loads(cj or "{}").get("quality", 0.8))
            except Exception:
                q = 0.8
            act, _ = self.add(task, code, video, q)
            kept += (act == "keep")
        return len(rows), kept


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
    p.add_argument("--skill", default=None, help="显式技能轴（生成端知道就传，避免正则猜错）")
    p.set_defaults(fn="add")
    p = sub.add_parser("sample"); p.add_argument("-k", type=int, default=5); p.set_defaults(fn="sample")
    p = sub.add_parser("stats"); p.set_defaults(fn="stats")
    p = sub.add_parser("backfill"); p.set_defaults(fn="backfill")
    a = ap.parse_args()

    arch = EliteArchive()
    if a.fn == "add":
        code_text = open(a.code, encoding="utf-8", errors="replace").read()
        act, why = arch.add(a.task, code_text, a.video, a.quality, skill=a.skill)
        print("%s %s" % (act.upper(), why))
    elif a.fn == "sample":
        for b, dom, sig, q, task in arch.sample(a.k):
            print("[%.2f] %s | %s | %s" % (q, dom, sig, (task or "")[:50]))
    elif a.fn == "stats":
        total, nb, buckets = arch.stats()
        print("精英总数 %d / 桶数 %d" % (total, nb))
        for dom, n, q in buckets:
            print("  %-12s %d 个, 均质 %.2f" % (dom, n, q or 0))
    elif a.fn == "backfill":
        n, kept = arch.rebuild_from_bank()
        total, nb, buckets = arch.stats()
        print("银行 %d 条 → 收精英 %d 条 / 桶数 %d" % (n, total, nb))
        for dom, cnt, q in buckets:
            print("  %-12s %d 个, 均质 %.2f" % (dom, cnt, q or 0))


if __name__ == "__main__":
    main()

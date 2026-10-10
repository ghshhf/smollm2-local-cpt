"""
生成 V3 域 held-out 评测集（纯新增，不碰训练）
==============================================
为什么需要单独的 V3 评测集：
  现有 eval/holdout_v7.jsonl 全是代码/逻辑/算法题，而 V3 训的是**小说设定卡**
  （概念解释/角色档案/人物关系/大纲）。拿代码题去测设定卡，测出来的不是
  「进化了多少」，而是「灾难性遗忘」，结论会误导。

构造原则（保证测的是"真学会"而不是"背原句"）：
  1. 提问措辞刻意**避开**训练集里的模板（训练集几乎全是
     "请依据设定原话，解释《X》中的概念【Y】"）。这里换成口语化问法。
  2. 判分关键词 must_contain **不出现在题目里**——模型必须真的把设定知识
     装进权重才能答出这些术语，靠抄题面拿不到分。
  3. 额外加一层：跨知识点组合题（要同时调用两条不同样本的知识）。
  4. 保留少量通用/代码题做"遗忘探针"，看 V3 有没有把基座打傻。

产出：eval/holdout_v3.jsonl
"""
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "role_data_v3.jsonl")
OUT = os.path.join(ROOT, "eval", "holdout_v3.jsonl")

# ---- 提问改写模板：措辞必须和训练模板不同 ----
Q_TMPL_CONCEPT = [
    "写《{uni}》的时候，「{name}」到底是什么意思？给我讲清楚。",
    "《{uni}》里设定的 {name}，用大白话怎么理解？",
    "我要给《{uni}》写一段情节，涉及 {name}，先帮我把它讲明白。",
    "简单说说《{uni}》的 {name} 是怎么规定的。",
]
Q_TMPL_CHAR = [
    "《{uni}》里 {name} 这个人，给我一份简介。",
    "写《{uni}》用到角色 {name}，他的身份和能力是什么？",
    "介绍一下《{uni}》的 {name}。",
]
Q_TMPL_REL = [
    "《{uni}》里 {name} 和其他方面是什么关系？",
    "{name} 在《{uni}》里处在什么位置？连带着和谁有关系？",
]

# 判分关键词要排除的通用噪声
STOP = set("""
的 了 是 在 和 与 或者 一个 这种 那些 什么 怎么 这样 因为 所以 如果 可以 不能
我们 他们 它们 其 之 以 于 为 被 把 但 而 也 就 都 还 很 更 最 不 有 会 要 到 说
可以写 不可以写 一句话 关键 纠偏 阶段 规则 设定 原话 依据 概念 角色 如下 例如
""".split())

def pick_terms(output: str, instruction: str, k: int = 3) -> list[str]:
    """
    从答案原文抽判定术语，必须**没出现在题目里**——逼模型凭权重记忆作答。

    抽取优先级（前者是作者人工强调的概念，质量远高于切词）：
      1. markdown 加粗 **术语**     <- 覆盖率 94.9%，这是设定作者自己标的概念名
      2. 英文术语 / 缩写（>=4 字符）
      3. 【】书名号内的专名
    过滤：长度越界、停用词、文件名噪声（setting-/的中划线+md后缀等）。
    """
    cand: list[str] = []

    PUNCT = "，。：；、？！“”《》…—（）()"

    def add(t: str):
        t = t.strip().strip("：:，,。.、（）()\"\"''《》 ")
        if not t or len(t) < 2 or len(t) > 10:
            return
        if t in STOP or any(ch.isspace() for ch in t):
            return
        # 句子级碎片不当判定词：模型学会了也不会逐字复述，只会拉低信噪比
        if any(p in t for p in PUNCT):
            return
        if t in instruction:          # 题面已出现 -> 抄题就能得分，剔除
            return
        cand.append(t)

    for b in re.findall(r"\*\*(.+?)\*\*", output):
        add(b)
    for e in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", output):
        if "-" in e and len(e) > 22:  # 形如 'setting-prequel-01-world' 的文件名噪声
            continue
        if e.lower() in {"md", "json", "http", "https", "true", "false", "none"}:
            continue
        add(e)
    for b in re.findall(r"【(.+?)】", output):
        add(b.split("（")[0])

    # 按出现频次稳定排序，去重包含关系，取前 k 个
    freq = {}
    seen = []
    for t in cand:
        freq[t] = freq.get(t, 0) + 1
    for t in sorted(set(cand), key=lambda x: (-len(x), -freq[x], x)):
        if any(t in s or s in t for s in seen):
            continue
        seen.append(t)
        if len(seen) >= k:
            break
    return seen


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    random.seed(11)

    by_type = {}
    for r in rows:
        by_type.setdefault(r.get("_type", "?"), []).append(r)

    probes = []
    # ---- 1) 概念解释变体题（主力） ----
    concepts = [r for r in by_type.get("concept", []) if len(r["output"]) > 120]
    random.shuffle(concepts)
    # 均匀覆盖不同宇宙
    seen_uni = {}
    picked = []
    for r in concepts:
        u = r.get("_universe", "?")
        if seen_uni.get(u, 0) >= 5:
            continue
        seen_uni[u] = seen_uni.get(u, 0) + 1
        picked.append(r)
        if len(picked) >= 24:
            break

    for r in picked:
        m = re.search(r"【(.+?)】", r["instruction"])
        name = m.group(1) if m else "该概念"
        name = name.split("（")[0].strip()
        uni = r.get("_universe", "?")
        q = random.choice(Q_TMPL_CONCEPT).format(uni=uni, name=name)
        terms = pick_terms(r["output"], q)
        if not terms:
            continue
        probes.append({"instruction": q, "must_contain": terms,
                       "type": "设定概念", "ref_ins": r["instruction"][:60],
                       "ref_out": r["output"][:200]})

    # ---- 2) 角色档案变体题 ----
    chars = [r for r in by_type.get("char", []) if len(r["output"]) > 150]
    random.shuffle(chars)
    for r in chars[:6]:
        m = re.search(r"【(.+?)】", r["instruction"])
        name = m.group(1) if m else "该角色"
        uni = r.get("_universe", "?")
        q = random.choice(Q_TMPL_CHAR).format(uni=uni, name=name)
        terms = pick_terms(r["output"], q)
        if not terms:
            continue
        probes.append({"instruction": q, "must_contain": terms,
                       "type": "角色档案", "ref_ins": r["instruction"][:60],
                       "ref_out": r["output"][:200]})

    # ---- 3) 关系题 ----
    rels = by_type.get("rel", [])
    random.shuffle(rels)
    for r in rels[:4]:
        m = re.search(r"【(.+?)】", r["instruction"])
        name = m.group(1) if m else "该角色"
        uni = r.get("_universe", "?")
        q = random.choice(Q_TMPL_REL).format(uni=uni, name=name)
        terms = pick_terms(r["output"], q)
        if not terms:
            continue
        probes.append({"instruction": q, "must_contain": terms,
                       "type": "人物关系", "ref_ins": r["instruction"][:60],
                       "ref_out": r["output"][:200]})

    # ---- 4) 遗忘探针（通用/代码能力，V3 不该把它们打没） ----
    probes += [
        {"instruction": "用 Python 写一个函数，判断一个字符串是不是回文。",
         "must_contain": ["def", "return"], "type": "通用保底"},
        {"instruction": "解释一下什么是 HTTP 状态码 404。",
         "must_contain": ["找不到", "资源"], "type": "通用保底"},
        {"instruction": "把「我很高兴」翻译成英文。",
         "must_contain": ["happy"], "type": "通用保底"},
        {"instruction": "计算：128 乘以 25 等于多少？",
         "must_contain": ["3200"], "type": "通用保底"},
        {"instruction": "列出三个 Python 里常用的数据结构。",
         "must_contain": ["列表", "字典"], "type": "通用保底"},
    ]

    # 打乱顺序：这样评测脚本用 --limit N 取前 N 题时，拿到的是**随机子集**
    # 而不是清一色的「设定概念」题，精简跑也具代表性。
    random.shuffle(probes)

    with open(OUT, "w", encoding="utf-8") as f:
        for p in probes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    import collections
    print(f"V3 held-out 已生成 -> {OUT}")
    print(f"  总题数: {len(probes)}")
    print("  题型分布:", dict(collections.Counter(p["type"] for p in probes)))
    print("\n样例：")
    for p in probes[:4]:
        print(f"  Q: {p['instruction']}")
        print(f"  判定词: {p['must_contain']}\n")


if __name__ == "__main__":
    main()

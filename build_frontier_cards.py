# -*- coding: utf-8 -*-
"""前沿论文 → 知识卡 v2：去重 + 细分主题 + 相关度分层。

改进（2026-10-17）:
  1. 跨列表去重：id 规范化（去 /vN 后缀）+ 标题精确匹配，95 唯一论文不重复建卡；
  2. 细分主题：10 个子主题，关键词规则分类，写 _topic；
  3. 相关度分层：core（直接服务北极星进化）/ related / low，写 _rel——
     巩固轮配比可按 _rel 加权（core 卡多采，low 卡少采或不采）。

输出:
  data/card_frontier.jsonl    知识卡（含 _topic/_rel）
  data/frontier/topics.md     按主题分组的索引（人读）
"""
import json, os, glob, io, sys, re, hashlib, collections

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "card_frontier.jsonl")
IDX = os.path.join(ROOT, "data", "frontier", "topics.md")

CAT_ZH = {"cs.CL": "语言模型", "cs.AI": "人工智能", "cs.LG": "机器学习",
          "cs.CV": "计算机视觉", "daily": "每日精选"}

# ---- 主题分类表：顺序即优先级，先命中先归箱 ----
TOPICS = [
    ("测试时优化与推理进化", r"test.?time|inference.?time|rejection sampl|prompt.*diffusion|EMO|context optimization"),
    ("持续学习与防遗忘", r"continual|forgetting|catastrophic|subspace tuning|replay|quality.?diversity"),
    ("记忆与经验管理", r"memory|experience|workspace|librarian|mosaic|multi.?session"),
    ("蒸馏与训练效率", r"distill|SFT|fine.?tun|policy gradient|credit assignment|reward.*token|LoRA"),
    ("模型调用工具", r"tool.?use|tool.?call|function call|scaffold|computer.?use|gui interaction|code execution|api.*agent|environment.?agnostic"),
    ("逆向工程", r"reverse engineer|decompil|disassembl|binary analysis|malware|patch diff"),
    ("多智能体与Agent治理", r"agent|multi.?agent|orchestr|swarm|weak.?to.?strong|routing|guard|safety|oversight|scaffold"),
    ("扩散与图像生成", r"diffusion|image gen|image edit|texture|caching.*transformer|text.?to.?image|merge.*token"),
    ("视觉语言与多模态", r"vision|VLM|multimodal|video|audio.?visual|caption|spatial|audio"),
    ("评测与基准", r"bench|evaluat|awareness|sensitivity|rubric|benchmark"),
    ("机器人与世界模型", r"robot|manipulation|world model|4D|dexterous|trajectory|motion forecast|quantum circuit"),
]
LOW_HINT = r"watermark|wind.?field|retinal|survival|vehicle|compression|theorem|bandit|coordinate descent|hardware|RAG|watermarking|triage|treatment|graph search|clinical|privacy"

def norm_id(pid):
    return re.sub(r"v\d+$", "", str(pid or ""))

def classify(title, cat):
    t = title.lower()
    for topic, pat in TOPICS:
        if re.search(pat, t):
            return topic, "core" if topic in ("测试时优化与推理进化", "持续学习与防遗忘", "记忆与经验管理", "蒸馏与训练效率") else "related"
    if re.search(LOW_HINT, t):
        return "其他（低相关）", "low"
    return "其他", "related"

def paper_cards(r, topic, rel):
    title = re.sub(r"\s+", " ", str(r.get("title") or "")).strip()
    abstract = re.sub(r"\s+", " ", str(r.get("summary") or r.get("abstract") or "")).strip()
    cat = r.get("cat", "")
    src = norm_id(r.get("id")) or hashlib.md5(title.encode()).hexdigest()[:12]
    if not title or not abstract:
        return []
    zh = CAT_ZH.get(cat, cat or "前沿")
    rel_zh = {"core": "核心", "related": "相关", "low": "低相关"}[rel]
    return [
        {"instruction": f"介绍一下论文《{title}》的核心内容",
         "output": f"【{topic}·{rel_zh}】这是{zh}方向的前沿工作（{src}）。摘要要点：{abstract}",
         "_type": "frontier_mech", "_src": src, "_topic": topic, "_rel": rel},
        {"instruction": f"《{title}》属于什么研究领域？",
         "output": f"属于{zh}领域下的「{topic}」子方向（{rel_zh}）。论文题目：{title}。",
         "_type": "frontier_field", "_src": src, "_topic": topic, "_rel": rel},
    ]

# ---- 扫描 + 双键去重（规范id / 标题） ----
papers = {}
n_in = n_dup = 0
for f in sorted(glob.glob(os.path.join(ROOT, "data", "frontier", "*.jsonl"))):
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_in += 1
        title = re.sub(r"\s+", " ", str(r.get("title") or "")).strip()
        key = norm_id(r.get("id")) or title
        tkey = title.lower()
        if key in papers or (tkey and tkey in {p[0].lower() for p in papers.values()}):
            n_dup += 1
            continue
        papers[key] = (title, r)

# ---- 生成卡片 ----
by_topic = collections.defaultdict(list)
n_out = 0
with open(OUT, "w", encoding="utf-8") as fo:
    for key, (title, r) in sorted(papers.items()):
        topic, rel = classify(title, r.get("cat", ""))
        for c in paper_cards(r, topic, rel):
            fo.write(json.dumps(c, ensure_ascii=False) + "\n")
            n_out += 1
        by_topic[topic].append((title, rel, r.get("cat", "")))

# ---- 主题索引 ----
with open(IDX, "w", encoding="utf-8") as fi:
    fi.write("# 前沿论文主题索引（自动生成于 build_frontier_cards.py，勿手改）\n\n")
    fi.write(f"扫描 {n_in} 行 → 去重 {n_dup} → 唯一论文 {len(papers)} 篇\n\n")
    for topic in sorted(by_topic, key=lambda k: -len(by_topic[k])):
        rows = by_topic[topic]
        n_core = sum(1 for _, rel, _ in rows if rel == "core")
        fi.write(f"## {topic}（{len(rows)} 篇，核心 {n_core}）\n")
        for title, rel, cat in sorted(rows):
            mark = {"core": "★", "related": "·", "low": "×"}[rel]
            fi.write(f"- {mark} [{cat}] {title}\n")
        fi.write("\n")

# ---- 控制台摘要 ----
print(f"扫描 {n_in} 行 → 去掉跨列表/版本重复 {n_dup} → 唯一论文 {len(papers)} 篇 → 卡片 {n_out} 条")
print("\n主题分布:")
for topic in sorted(by_topic, key=lambda k: -len(by_topic[k])):
    rows = by_topic[topic]
    n_core = sum(1 for _, rel, _ in rows if rel == "core")
    print(f"  {topic:14s} {len(rows):3d} 篇（核心 {n_core}）")
print(f"\n索引 -> {os.path.relpath(IDX, ROOT)}")

# -*- coding: utf-8 -*-
"""
前沿论文分类报告：data/frontier/*.jsonl -> 分类汇总 + 自包含 HTML。
- 按标题+摘要关键词分类，规则顺序=优先级（与本地六能力线相关的排前面）
- 核心标注：与 本地线（推理时学习/记忆/持续学习/蒸馏效率/识图/音频）直接相关的论文
- 输出: docs/frontier-crawl-report-<date>.html
"""
import json
import glob
import re
import time
from collections import OrderedDict, Counter

# (类别名, [关键词正则], 是否核心相关) —— 顺序即优先级，首个命中即归属
RULES = [
    ("推理时学习/TTT", r"test[- ]time|fast weight|ttt|in[- ]context (weight|learning)|learning at inference|test[- ]time train", "core"),
    ("记忆与经验", r"memory|experience (replay|bank)|episodic|knowledge edit", "core"),
    ("持续学习/防遗忘", r"continual|catastrophic forgetting|lifelong|forgetting", "core"),
    ("蒸馏/训练效率/小模型", r"distill|small (language )?model|efficient (training|inference)|pruning|quantiz|sparse (auto|attention)|data (selection|curation)|sft|fine[- ]tun", "core"),
    ("RL/自进化", r"reinforcement|reward|self[- ](improv|evol|play|generat)|policy optimization|rsi|curriculum", "evo"),
    ("视觉/多模态", r"vision|visual|multimodal|multi[- ]modal|vlm|image|video understand|caption|grounding|segment", None),
    ("音频/语音", r"\baudio\b|speech|asr|whisper|acoustic|voice", None),
    ("代码/编程", r"code generation|programming|software engineer|\bswe\b|repository|coding agent", None),
    ("Agent/检索/RAG", r"\bagent|agentic|retrieval|\brag\b|search (agent|engine)|tool use|orchestrat|web ", None),
    ("生成模型", r"diffusion|text[- ]to[- ]image|image generation|generative model|video generation|gan\b", None),
    ("评测/基准", r"benchmark|evaluation|evaluating|\bevals?\b", None),
    ("机器人/世界模型", r"robot|manipulation|world model|embodied|navigation", None),
    ("安全/对齐", r"safety|alignment|jailbreak|honesty|hallucinat|watermark", None),
    ("长文本/推理链", r"long context|chain[- ]of[- ]thought|reasoning|chain of thought", None),
]

def classify(text):
    tl = text.lower()
    for name, pat, tier in RULES:
        if re.search(pat, tl):
            return name, tier
    return "其他", None

def main():
    # 1. 载入 + 按 id 去重
    papers = OrderedDict()
    files = sorted(glob.glob("data/frontier/2026-*.jsonl"))
    per_file = {}
    for fp in files:
        n = 0
        for ln in open(fp, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            n += 1
            papers[d["id"]] = d
        per_file[fp] = n
    for p in papers.values():
        name, tier = classify(p.get("title", "") + " " + p.get("abstract", "")[:600])
        p["_cat"] = name
        p["_tier"] = tier

    cats = OrderedDict()
    for p in papers.values():
        cats.setdefault(p["_cat"], []).append(p)
    # 排序：核心相关类别在前，桶内核心在前
    order = [r[0] for r in RULES] + ["其他"]
    cat_sorted = sorted(cats.items(), key=lambda kv: (order.index(kv[0]), 0))
    for name, plist in cat_sorted:
        plist.sort(key=lambda p: ({"core":0,"evo":1}.get(p["_tier"], 2), p.get("title", "")))

    total = len(papers)
    n_core = sum(1 for p in papers.values() if p["_tier"] == "core")
    n_evo = sum(1 for p in papers.values() if p["_tier"] == "evo")

    # 2. HTML
    now = time.strftime("%Y-%m-%d %H:%M")
    rows = []
    for cname, plist in cat_sorted:
        n_c = sum(1 for p in plist if p["_tier"] == "core")
        n_e = sum(1 for p in plist if p["_tier"] == "evo")
        tag = f'（{len(plist)} 篇，★现役核心 {n_c}' + (f'，◆自进化储备 {n_e}' if n_e else '') + '）'
        items = []
        for p in plist[:40]:
            if p["_tier"] == "core":
                mark, cls = "★ ", ' style="color:#c0392b;font-weight:bold"'
            elif p["_tier"] == "evo":
                mark, cls = "◆ ", ' style="color:#8e6d1f"'
            else:
                mark, cls = "", ""
            abstract = (p.get("abstract") or "")[:150].replace("<", "&lt;")
            items.append(
                f'<li>{mark}<span{cls}>[{p.get("cat")}] '
                f'<a href="{p.get("url")}" target="_blank">{p.get("title","")[:110]}</a></span>'
                f'<br><small style="color:#667">{abstract}…</small></li>')
        more = f'<li><small>…另有 {len(plist)-40} 篇</small></li>' if len(plist) > 40 else ""
        rows.append(
            f'<h3>{cname}{tag}</h3><ul>{"".join(items)}{more}</ul>')

    file_rows = "".join(
        f"<tr><td>{fp}</td><td>{n}</td></tr>" for fp, n in per_file.items())
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>前沿论文爬取分类报告 {now}</title>
<style>body{{font-family:'Microsoft YaHei',sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#223;line-height:1.55}}
h1{{font-size:22px;border-bottom:3px solid #3465a4;padding-bottom:8px}} h3{{margin:18px 0 6px;color:#3465a4;border-left:4px solid #3465a4;padding-left:8px}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ccd;padding:4px 12px;text-align:left}}
.kpi{{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0}}.kpi div{{background:#eef3fa;border-radius:8px;padding:10px 18px}}
.kpi b{{font-size:26px;color:#3465a4;display:block}} ul{{padding-left:20px;margin:6px 0}} li{{margin:6px 0}} a{{color:#246;text-decoration:none}} a:hover{{text-decoration:underline}}
small{{color:#667}}</style></head><body>
<h1>前沿论文爬取·分类报告</h1>
<p>生成时间 {now} ｜ 数据源 arXiv(cs.AI/LG/CL/CV) + HuggingFace Daily Papers ｜ 每小时自动抓取（FrontierCrawl）</p>
<div class="kpi">
<div><b>{total}</b>唯一论文(去重后)</div><div><b>{n_core}</b>★现役核心(本地线)</div><div><b>{n_evo}</b>◆自进化储备</div>
<div><b>{len(cat_sorted)}</b>分类桶</div><div><b>{len(files)}</b>归档日文件</div>
</div>
<h3>爬取台账</h3><table><tr><th>归档文件</th><th>条数</th></tr>{file_rows}</table>
<p><small>去重说明：按 arXiv id 全局去重；跨日重复（如 10-15 回填与 10-17 重叠）只算一篇。每小时一轮 5 源×30 条，命中已见过的自动跳过，日志见 logs/frontier_crawl.log。</small></p>
<h3>分类明细</h3><p><small>★红=现役核心（推理时学习/记忆/持续学习/蒸馏效率，直接喂本地线）｜◆金=自进化储备（RL/做梦/三元组银行/RSI 线未来主粮）｜黑=一般参考</small></p>
{"".join(rows)}
</body></html>"""
    out = f"docs/frontier-crawl-report-{time.strftime('%Y-%m-%d')}.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    # 3. 控制台摘要
    print(f"唯一论文 {total} 篇（去重前 {sum(per_file.values())}），现役核心 {n_core} 篇，自进化储备 {n_evo} 篇")
    for cname, plist in cat_sorted:
        n_c = sum(1 for p in plist if p["_tier"] == "core")
        n_e = sum(1 for p in plist if p["_tier"] == "evo")
        extra = f" (★核心 {n_c}" + (f"，◆储备 {n_e}" if n_e else "") + ")" if (n_c or n_e) else ""
        print(f"  {cname}: {len(plist)} 篇{extra}")
    print("报告:", out)

if __name__ == "__main__":
    main()

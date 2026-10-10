# -*- coding: utf-8 -*-
"""
build_dataset_v4.py
V4 = 事实(v3 继承, 防遗忘) + 成稿叙事(新主料) + 代码理解(用户指定的编程料)
- 事实: 直接继承 role_data_v3.jsonl(624 条), 不重复造
- 成稿: 只抽 novels/<宇宙>/第N章.md(排除 meta/state/大纲/原稿), 两种样本:
    prose_style : 风格brief -> 一段叙事(教 voice)
    prose_cont  : 前缀片段 -> 续写(教 generation, 强制生成不背课文)
- 代码: 从 distill_code_dataset.jsonl 采样 200 条(指令->代码), 教"理解/写代码"
全局精确去重(指令+输出). 输出 data/role_data_v4.jsonl
"""
import json, re, os, glob, random

random.seed(20261007)
NOVELS   = "D:/work/novel/novels"
CODE_DS  = "D:/tgapp/telegram_mini_app/distill_code_dataset.jsonl"
V3       = "D:/work/_local_ai/train_15b/data/role_data_v3.jsonl"
OUT      = "D:/work/_local_ai/train_15b/data/role_data_v4.jsonl"

EXCLUDE_FILES = {"meta.json", "state.md", "STATUS.md", "大纲.md"}
PROSE_TARGET  = 460   # 成稿样本上限(按宇宙均衡后随机抽取)
CODE_TARGET   = 200   # 代码样本上限

def load_jsonl(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# ---------- 1. 继承 v3 事实 ----------
v3 = load_jsonl(V3)
print(f"[v3] 继承事实样本: {len(v3)}")

# ---------- 2. 抽取成稿叙事 ----------
def get_brief(ud):
    mp = os.path.join(ud, "meta.json")
    if os.path.exists(mp):
        try:
            m = json.load(open(mp, encoding="utf-8"))
            b = m.get("system", "") or m.get("genre", "") or ""
            return b.strip()
        except Exception:
            return ""
    return ""

prose_candidates = []
universe_counts = {}
for u in sorted(os.listdir(NOVELS)):
    ud = os.path.join(NOVELS, u)
    if not os.path.isdir(ud):
        continue
    brief = get_brief(ud)[:200]
    chapters = []
    for f in sorted(glob.glob(os.path.join(ud, "*.md"))):
        base = os.path.basename(f)
        if base in EXCLUDE_FILES or "原稿" in base:
            continue
        txt = open(f, encoding="utf-8").read()
        chapters.append((base, txt))
    for fname, txt in chapters:
        paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
        paras = [p for p in paras if len(p) > 30]
        if len(paras) < 2:
            continue
        # Type A: 风格 brief -> 一段叙事
        for i in range(0, len(paras) - 1, 2):
            out = "\n\n".join(paras[i:i + 2])
            if 120 <= len(out) <= 760:
                instr = (f"你是一位小说写手。{brief}\n"
                         f"请以以上风格与世界观，创作一段叙事文本。")
                prose_candidates.append({
                    "instruction": instr, "output": out,
                    "_type": "prose_style", "_universe": u, "_src": fname})
        # Type B: 前缀 -> 续写(强制生成)
        for i in range(0, len(paras) - 2):
            prefix, cont = paras[i], paras[i + 1]
            if len(prefix) < 60 or len(cont) < 60:
                continue
            if len(cont) > 760:
                cont = cont[:760]
            instr = (f"请续写以下「{u}」风格的小说片段，保持笔调、视角与世界观一致：\n\n"
                     f"{prefix[:500]}")
            prose_candidates.append({
                "instruction": instr, "output": cont,
                "_type": "prose_cont", "_universe": u, "_src": fname})
    universe_counts[u] = len(chapters)

print(f"[prose] 候选成稿样本: {len(prose_candidates)}  来自宇宙: {universe_counts}")

# 按宇宙均衡抽样到 PROSE_TARGET
by_u = {}
for r in prose_candidates:
    by_u.setdefault(r["_universe"], []).append(r)
sampled = []
per_u = max(1, PROSE_TARGET // max(1, len(by_u)))
for u, rs in by_u.items():
    random.shuffle(rs)
    sampled.extend(rs[:per_u])
if len(sampled) < PROSE_TARGET:
    leftover = [r for r in prose_candidates if r not in sampled]
    random.shuffle(leftover)
    sampled.extend(leftover[:PROSE_TARGET - len(sampled)])
random.shuffle(sampled)
prose_rows = sampled[:PROSE_TARGET]
print(f"[prose] 抽样后成稿样本: {len(prose_rows)}")

# ---------- 3. 代码理解样本 ----------
code_rows = []
if os.path.exists(CODE_DS):
    allc = load_jsonl(CODE_DS)
    # 只取输出长度适中(<=720)的, 避免截断难看
    ok = [c for c in allc if 40 <= len(c.get("output", "")) <= 720
          and 20 <= len(c.get("instruction", "")) <= 400]
    random.shuffle(ok)
    code_rows = [{
        "instruction": c["instruction"].strip(),
        "output": c["output"].strip(),
        "_type": "code", "_universe": "code", "_src": c.get("_seed", "distill")
    } for c in ok[:CODE_TARGET]]
print(f"[code] 代码样本: {len(code_rows)} (源 {len(allc)} 条, 过滤后 {len(ok)} 条可用)")

# ---------- 4. 合并 + 全局精确去重 ----------
seen = set()
final = []
for r in v3 + prose_rows + code_rows:
    key = (r["instruction"].strip(), r["output"].strip())
    if key in seen:
        continue
    seen.add(key)
    final.append(r)

from collections import Counter
tc = Counter(r["_type"] for r in final)
uc = Counter(r["_universe"] for r in final)
print(f"[merge] 去重后总数: {len(final)}")
print(f"[merge] 按类型: {dict(tc)}")
print(f"[merge] 按宇宙(top12): {dict(uc.most_common(12))}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in final:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[write] -> {OUT}")

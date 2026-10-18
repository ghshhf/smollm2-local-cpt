#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_jsonl.py — 语料进场体检（先读懂，再谈精炼）
用法: venv/Scripts/python.exe tools/inspect_jsonl.py <文件> [抽样条数]
支持: .jsonl / .jsonl.zst(需zstandard) / 首行为数组的 .json
输出: 字段结构 / 长度分布 / 重复检测 / 噪声特征 / 原始样本
"""
import sys, os, json, collections, re

path = sys.argv[1]
nsamp = int(sys.argv[2]) if len(sys.argv) > 2 else 8
MAXROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 60000

print("=" * 78)
print("语料体检:", path)
print("=" * 78)
print("[size] %.1f MB" % (os.path.getsize(path) / 1e6))

CJK = re.compile(r"[\u4e00-\u9fff]")

def pick_text(r):
    if isinstance(r, str):
        return r
    for k in ("text", "content", "body", "article", "output"):
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # 兜底: 取最长的字符串字段
    best = ""
    for k, v in r.items():
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best

rows = []
bad = 0
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i >= MAXROWS:
            break
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            bad += 1

print("[rows scanned] %d  (坏行 %d)" % (len(rows), bad))
if not rows:
    print("没有读到数据，退出"); sys.exit(1)

keys = collections.Counter(tuple(sorted(r.keys())) for r in rows if isinstance(r, dict))
print("[字段组合 top3]")
for k, v in keys.most_common(3):
    print("   %-60s %d" % (str(k)[:60], v))

texts = [pick_text(r) for r in rows]
titles = [r.get("title") for r in rows if isinstance(r, dict) and r.get("title")]
if titles:
    tc = collections.Counter(titles)
    print("[title] 唯一书目 %d / 样本 %d" % (len(tc), len(titles)))
    for k, v in tc.most_common(5):
        print("   %-40s %d" % (str(k)[:40], v))

lens = sorted(len(t) for t in texts)
print("\n[长度] min=%d p10=%d p25=%d med=%d p75=%d p90=%d p99=%d max=%d mean=%d"
      % (lens[0], lens[len(lens)//10], lens[len(lens)//4], lens[len(lens)//2],
         lens[len(lens)*3//4], lens[len(lens)*9//10], lens[int(len(lens)*.99)],
         lens[-1], sum(lens)//len(lens)))
tot = sum(lens)
nbytes = os.path.getsize(path)
# 全量外推
print("  样本 %d 行共 %d 字符 → 全文件约 %.2f 亿字符" % (len(texts), tot, tot / len(texts) * (nbytes / (nbytes and 1)) / 1e8) if False else
      "  样本 %d 行共 %d 字符，均值 %d" % (len(texts), tot, tot // len(texts)))
han = sum(len(CJK.findall(t)) for t in texts)
print("  汉字数 %d (占比 %.1f%%)，按 2.4589 tok/汉字估 → 样本约 %.2f 亿 token"
      % (han, han * 100.0 / max(1, tot), han * 2.4589 / 1e8))

# 重复检测
fp = collections.Counter(t[:100] for t in texts if len(t) >= 100)
dup = sum(v - 1 for v in fp.values() if v > 1)
base = sum(1 for t in texts if len(t) >= 100)
print("\n[重复] 前100字指纹 唯一 %d / %d，重复 %d (%.2f%%)"
      % (len(fp), base, dup, dup * 100.0 / max(1, base)))
top = [(k, v) for k, v in fp.most_common(3) if v > 1]
for k, v in top:
    print("   x%d  %s" % (v, k.replace("\n", " ")[:70]))

def _listy(t):
    ls = [x for x in t.split("\n") if x.strip()]
    if len(ls) < 4: return False
    n = sum(1 for x in ls if re.match(r"^\s{0,4}([-*•]|\d+[.、)])", x))
    return n / len(ls) > 0.3

checks = {
    "HTML标签": lambda t: bool(re.search(r"<[a-zA-Z/][^>]{0,40}>", t)),
    "URL链接": lambda t: bool(re.search(r"https?://|www\.", t)),
    "markdown标题": lambda t: bool(re.search(r"^\s{0,3}#{1,6}\s", t, re.M)),
    "代码块```": lambda t: "```" in t,
    "广告/推广词": lambda t: bool(re.search(r"(欢迎收藏|求推荐票|求月票|本书首发|请关注微信|手机阅读|起点中文网|笔趣阁|小说免费阅读)", t)),
    "色情/成人征兆": lambda t: bool(re.search(r"(娇喘|呻吟|内裤|胸部|乳房|呻吟|淫|肉棒|阴道|做爱)", t)),
    "纯列表体": _listy,
    "对话占比>50%": lambda t: t.count("“") + t.count('"') > 0 and (t.count("“") + t.count('"')) / max(1, len(t)) > 0.02,
    "结尾截断": lambda t: bool(t) and t.strip()[-1] not in "。！？!?.…”』）】》\"'",
    "短于100字": lambda t: len(t) < 100,
}
print("\n[噪声/特征体检] (样本 %d 行)" % len(texts))
for name, fn in checks.items():
    c = sum(1 for t in texts if t and fn(t))
    print("   %-22s %7d  (%.2f%%)" % (name, c, c * 100.0 / len(texts)))

print("\n" + "=" * 78)
print("原始样本 (%d 条，每条 600 字)" % nsamp)
print("=" * 78)
step = max(1, len(texts) // nsamp)
for i in range(0, min(len(texts), nsamp * step), step):
    t = texts[i]
    r = rows[i]
    head = ""
    if isinstance(r, dict) and r.get("title"):
        head = "【%s】%s" % (r.get("title"), r.get("chapter", ""))
    print("\n----- idx=%d len=%d %s -----" % (i, len(t), head[:60]))
    print(t[:600])
    if len(t) > 600: print("  ...[共%d字]" % len(t))

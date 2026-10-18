#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
l3_inspect.py — 在动手精炼前，先把 L3 multi_style 数据看明白
用法: venv/Scripts/python.exe tools/l3_inspect.py [片号] [抽样条数]
输出: 字段结构 / 长度分布 / style 分布 / 重复检测 / 原始样本
"""
import sys, os, collections, re, json
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "l3_ms")

part = sys.argv[1] if len(sys.argv) > 1 else "00000"
nsamp = int(sys.argv[2]) if len(sys.argv) > 2 else 8
path = os.path.join(DATA, "ms%s.parquet" % part)

print("=" * 78)
print("L3 数据体检:", path)
print("=" * 78)

pf = pq.ParquetFile(path)
print("[schema]\n%s" % pf.schema)
nrows = pf.metadata.num_rows
print("[rows] %d" % nrows)
print("[row_groups] %d" % pf.metadata.num_row_groups)

# 只读前 20 万行做体检 + 抽样（省内存省时间）
SAMPLE_N = min(200000, nrows)
tbl = pf.read_row_group(0) if pf.metadata.num_row_groups >= 1 else pf.read()
cols = tbl.column_names
print("[cols] %s" % cols)
if tbl.num_rows > SAMPLE_N:
    tbl = tbl.slice(0, SAMPLE_N)

# 提取文本
def pick_text(row):
    for k in ("content", "text", "response", "output"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    q, a = row.get("question"), row.get("answer")
    if isinstance(q, str) and q.strip():
        return (q + "\n" + (a or "")) if isinstance(a, str) else q
    return ""

rows = []
for i in range(tbl.num_rows):
    row = {c: tbl.column(c)[i].as_py() for c in cols}
    rows.append(row)

texts = [pick_text(r) for r in rows]
lens = sorted(len(t) for t in texts)
empty = sum(1 for t in texts if not t.strip())

print("\n[文本长度] 样本 %d 行, 空行 %d (%.2f%%)" % (len(texts), empty, empty * 100.0 / max(1, len(texts))))
if lens:
    def pct(p): return lens[min(len(lens) - 1, int(len(lens) * p))]
    print("  min=%d  p10=%d  p25=%d  med=%d  p75=%d  p90=%d  p99=%d  max=%d  mean=%d"
          % (lens[0], pct(.10), pct(.25), pct(.50), pct(.75), pct(.90), pct(.99), lens[-1],
             sum(lens) // len(lens)))
    # token 估算: 中文按 2.4589 tok/汉字(经验值), 这里按字符粗估
    tot_char = sum(lens)
    print("  样本总字符 %d  → 全片约 %.1f 亿字符" % (tot_char, tot_char / SAMPLE_N * nrows / 1e8))

# style 分布
for key in ("style", "type", "category", "source", "uid"):
    if key in cols:
        c = collections.Counter(str(r.get(key))[:40] for r in rows)
        print("\n[%s] 唯一值 %d, top10:" % (key, len(c)))
        for k, v in c.most_common(10):
            print("   %-42s %7d (%.2f%%)" % (k, v, v * 100.0 / len(rows)))

# 重复检测: 前 80 字指纹
fp = collections.Counter(t[:80] for t in texts if len(t) >= 80)
dup = sum(v - 1 for v in fp.values() if v > 1)
print("\n[重复检测] 前80字指纹: 唯一 %d / 总 %d, 重复行 %d (%.2f%%)"
      % (len(fp), sum(1 for t in texts if len(t) >= 80), dup,
         dup * 100.0 / max(1, sum(1 for t in texts if len(t) >= 80))))
top_dup = [(k, v) for k, v in fp.most_common(3) if v > 1]
if top_dup:
    print("  最高重复样例:")
    for k, v in top_dup:
        print("    x%d  %s" % (v, k.replace("\n", " ")[:70]))

# 噪声特征体检
checks = {
    "含HTML标签": lambda t: bool(re.search(r"<[a-zA-Z/][^>]{0,40}>", t)),
    "含URL": lambda t: bool(re.search(r"https?://|www\.", t)),
    "含markdown标题": lambda t: bool(re.search(r"^\s{0,3}#{1,6}\s", t, re.M)),
    "含代码块```": lambda t: "```" in t,
    "含英文长段(>40连续字母词)": lambda t: bool(re.search(r"[A-Za-z]{40,}", t)),
    "含'作为AI'类自述": lambda t: bool(re.search(r"作为(一个)?(AI|人工智能|语言模型|助手)", t)),
    "含模板套话(首先/其次/总之)": lambda t: bool(re.search(r"(首先|其次|再次|最后|总之|综上所述)", t)),
    "纯列表(行首-或数字.占比>30%)": lambda t: _is_listy(t),
    "结尾截断(无句号/感叹/问号结尾)": lambda t: bool(t) and t.strip()[-1] not in "。！？!?.…)）】》\"'",
}

def _is_listy(t):
    ls = [x for x in t.split("\n") if x.strip()]
    if len(ls) < 4: return False
    n = sum(1 for x in ls if re.match(r"^\s{0,4}([-*•]|\d+[.、)])", x))
    return n / len(ls) > 0.3

print("\n[噪声特征体检] (样本 %d 行)" % len(texts))
for name, fn in checks.items():
    try:
        c = sum(1 for t in texts if t and fn(t))
    except Exception as e:
        c = -1
    print("   %-28s %7d  (%.2f%%)" % (name, c, c * 100.0 / max(1, len(texts))))

# 原始样本
print("\n" + "=" * 78)
print("原始样本 (%d 条, 每条截取前 700 字)" % nsamp)
print("=" * 78)
step = max(1, len(texts) // nsamp)
for i in range(0, min(len(texts), nsamp * step), step):
    t = texts[i]
    print("\n----- idx=%d  len=%d -----" % (i, len(t)))
    print(t[:700])
    if len(t) > 700: print("  ...[截断, 共%d字]" % len(t))

"""查看编程语料"同问多答"的真实差异：是同一答案的变体，还是真矛盾。"""
import collections
import difflib
import json
import sys

path = sys.argv[1]
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
byi = collections.defaultdict(list)
for r in rows:
    byi[r["instruction"].strip()].append(r["output"].strip())

conf = {k: v for k, v in byi.items() if len(v) > 1}
print(f"唯一问题 {len(byi)}  冲突组 {len(conf)}  多余条数 {sum(len(v)-1 for v in conf.values())}")

buckets = {"几乎相同(>0.9)": 0, "高度相似(0.7-0.9)": 0, "部分相似(0.4-0.7)": 0, "基本不同(<0.4)": 0}
samples = {k: [] for k in buckets}
for q, outs in conf.items():
    a, b = outs[0], outs[1]
    r = difflib.SequenceMatcher(None, a, b).ratio()
    if r > 0.9:
        k = "几乎相同(>0.9)"
    elif r > 0.7:
        k = "高度相似(0.7-0.9)"
    elif r > 0.4:
        k = "部分相似(0.4-0.7)"
    else:
        k = "基本不同(<0.4)"
    buckets[k] += 1
    if len(samples[k]) < 2:
        samples[k].append((q, a, b, r))

for k, v in buckets.items():
    print(f"  {k}: {v} 组 ({v/len(conf)*100:.1f}%)")

for k, items in samples.items():
    for q, a, b, r in items[:1]:
        print(f"\n===== {k}  ratio={r:.3f}  问题: {q[:50]}")
        print("  A1:", a.replace("\n", " ")[:110])
        print("  A2:", b.replace("\n", " ")[:110])

# 三答以上的组
more = [(q, v) for q, v in conf.items() if len(v) > 2]
print(f"\n三答及以上: {len(more)} 组")
for q, v in more[:2]:
    print(f"  {q[:50]} -> {len(v)} 答")

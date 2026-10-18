#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并 V1-02 的 6 个 alpha 分片结果 -> 完整阶梯 JSON（含裁决）"""
import glob, json, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "v1_02_alpha_ladder.json"
PAT = sys.argv[2] if len(sys.argv) > 2 else "v1_02_shard_*.json"
D = os.path.dirname(os.path.abspath(OUT))
shards = sorted(glob.glob(os.path.join(D, PAT)))
if not shards:
    print("NO_SHARDS")
    sys.exit(1)

rows, ce_map, base, tokens = [], {}, None, None
for s in shards:
    d = json.load(open(s, encoding="utf-8"))
    ce_map.update(d.get("ce_map", {}))
    tokens = d.get("tokens") or tokens
    for r in d.get("rows", []):
        if r.get("alpha") is None:
            base = r["ce"]
        else:
            rows.append(r)

by_a = {r["alpha"]: r for r in rows}
ALPHAS = sorted(by_a.keys(), reverse=True)   # 由实际分片决定，支持任意 alpha 列表
ordered = ([dict(alpha=None, ce=base, label="fp32 baseline")] if base is not None else [])
ordered += [by_a[a] for a in ALPHAS if a in by_a]

pure = ce_map.get("0.0", base)
delta_pure = (pure - base) if (pure is not None and base is not None) else None
verdicts = {}
if delta_pure is not None:
    verdicts = {
        "static_shock_small(d<0.3)": delta_pure < 0.3,
        "static_shock_moderate(0.3<=d<1.2)": 0.3 <= delta_pure < 1.2,
        "static_shock_large(d>=1.2)": delta_pure >= 1.2,
        "monotonic_in_alpha": all(
            ce_map.get(f"{ALPHAS[i]}", 1e9) >= ce_map.get(f"{ALPHAS[i+1]}", -1e9) - 1e-6
            for i in range(len(ALPHAS) - 1)
        ),
    }

res = dict(rows=ordered, ce_map={k: round(v, 4) for k, v in ce_map.items()},
           verdicts=verdicts,
           delta_pure=(round(delta_pure, 4) if delta_pure is not None else None),
           model="SmolLM2-135M (inherit, NO training)",
           tokens=tokens, shards=len(shards),
           note="alpha 分片并行后合并")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

print("merged %d shards -> %s" % (len(shards), OUT))
print("fp32 CE=%s" % base)
for a in ALPHAS:
    if a in by_a:
        print("alpha=%.1f CE=%.4f delta=%+.4f" % (a, by_a[a]["ce"], by_a[a]["delta"]))
print("delta_pure=%s" % (round(delta_pure, 4) if delta_pure is not None else None))
for k, v in verdicts.items():
    print("  %s %s" % ("PASS" if v else "no  ", k))

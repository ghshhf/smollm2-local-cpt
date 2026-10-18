"""导出可开源的编程模块训练数据。

原则：训练用全量，开源只放脱敏子集。差异在 README 里明说，不假装一样。

剔除的是「真实项目业务细节」：配置字段名、策略概念、具体标的。
保留的是「教学示例占位符」：api_key: ""、127.0.0.1、git@github.com:user/repo.git、
D:\\app\\run.bat —— 这些是任何人都会写的示例，不构成隐私。
"""
import json
import io
import re
import argparse
from collections import Counter

# 真实项目符号：命中即剔除整条
PRIVATE_SYMBOLS = [
    # 项目名
    "crypto_stocks", "millionFund", "LocalSend", "picgo", "skynet",
    # 配置与业务概念（用户量化系统的私有设计）
    "strategy_config", "weekly_theme", "REGIME_ALLOC", "event_override",
    "elastic_weights", "fallback_offensive", "fallback_defensive",
    "strict_unknown_lot", "tradable_markets", "ai_overlay",
    "market_of", "_elastic_score", "load_event_override",
    # 具体标的
    "凯莱英", "长江电力",
]

# 人格词：属于模块能力，不是隐私；保留但统计
PERSONA = ["北极星", "主人"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in io.open(a.inp, encoding="utf-8")]
    keep, drop = [], Counter()
    reasons = {}

    for i, r in enumerate(rows):
        t = r["text"]
        hit = [s for s in PRIVATE_SYMBOLS if s in t]
        if hit:
            drop["私有项目符号"] += 1
            reasons.setdefault("私有项目符号", []).append((i, hit[:3]))
            continue
        # 空壳 / 过短
        comp = t.split("<|im_start|>assistant\n")[-1]
        if len(comp.strip()) < 30:
            drop["completion过短"] += 1
            continue
        if "```" not in comp:
            drop["无代码块"] += 1
            continue
        keep.append(r)

    io.open(a.out, "w", encoding="utf-8").write(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))

    print(f"输入 {len(rows)} 条")
    for k, v in drop.items():
        print(f"  剔除 {k}: {v}")
    print(f"输出 {len(keep)} 条 -> {a.out}")
    print(f"\n私有符号命中示例（前 10 条）：")
    for i, hs in reasons.get("私有项目符号", [])[:10]:
        print(f"  [{i}] {hs}")


if __name__ == "__main__":
    main()

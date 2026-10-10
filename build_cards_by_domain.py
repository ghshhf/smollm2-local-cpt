"""
领域专卡生成器：从「累积继承」改为「专卡专用、用时才合」
==========================================================
背景（用户 2026-10-08 裁定）：
  旧模式 V3⊂V4⊂V5 是数据累积继承，每版数据集越来越大 -> 训练时间线性膨胀，不合理。
  新模式：每张卡专注一个领域，卡之间内容**不重叠**，训练时各自独立从基座训，
          使用的时候再按需切换或合并（见 cards/mount.py / cards/merge_card.py）。

处理动作：
  1. 汇总所有现存 jsonl，按 _type 归入领域桶（novel / prose / code / reason）
  2. 桶内按 instruction 去重（同时对完全相同的 instruction+output 也去重）
  3. 各卡互不重叠 -> 训练集不再虚胖，每张卡可独立训、独立迭代
  4. 输出 data/card_<domain>.jsonl

领域划分依据（来自实际数据分布）：
  novel  设定百科  = concept / char / rel / outline / rules / seam
  prose  文风      = prose_style / prose_cont（含新作的风格样本）
  code   编程      = code / code_fn / code_distill / code_paradigm
  reason 推理      = reason_cot（取 V8 结构化版：过程标签 + 思考/答案边界令牌）
"""
import collections
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

SOURCES = ["role_data_v3.jsonl", "role_data_v4.jsonl", "role_data_v5.jsonl",
           "role_data_v6.jsonl", "role_data_v7_selftaught.jsonl",
           "role_data_v8.jsonl"]

DOMAIN_OF_TYPE = {
    "concept": "novel", "char": "novel", "rel": "novel",
    "outline": "novel", "rules": "novel", "seam": "novel",
    "prose_style": "prose", "prose_cont": "prose",
    "code": "code", "code_fn": "code", "code_distill": "code", "code_paradigm": "code",
    "reason_cot": "reason",
}

DOMAIN_DESC = {
    "novel": "小说设定百科（概念/角色/关系/大纲/规则/接缝）",
    "prose": "行文风格与叙事延续（含新作样本）",
    "code": "编程（函数/知识蒸馏/范式）",
    "reason": "推理（过程标签 + 思考/答案边界）",
}


def load(path):
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path, encoding="utf-8"):
        s = l.strip()
        if s:
            out.append(json.loads(s))
    return out


def main():
    buckets = collections.defaultdict(list)
    total_rows = 0
    for name in SOURCES:
        rows = load(os.path.join(DATA, name))
        if not rows:
            continue
        total_rows += len(rows)
        for r in rows:
            d = DOMAIN_OF_TYPE.get(r.get("_type", ""))
            # 未识别类型直接跳过，避免脏数据混进任何一张卡
            if d is None:
                continue
            buckets[d].append(r)
        print(f"  读取 {name}: {len(rows)} 行")

    print(f"\n源文件总行数 {total_rows}\n")

    summary = []
    for d in ("novel", "prose", "code", "reason"):
        rows = buckets.get(d, [])
        seen = set()
        dedup = []
        for r in rows:
            key = r["instruction"].strip()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(r)
        # 稳定排序，保证每次生成结果一致（可复现）
        dedup.sort(key=lambda r: r["instruction"])
        out_path = os.path.join(DATA, f"card_{d}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in dedup:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary.append((d, len(rows), len(dedup)))
        print(f"card_{d:6s} {DOMAIN_DESC[d]}")
        print(f"    {len(rows)} 行 -> 去重 {len(dedup)} 条 (剔除 {len(rows)-len(dedup)})  -> {out_path}")

    print("\n=== 各卡互不重叠，可各自独立训练 ===")
    tot_src = sum(s[1] for s in summary)
    tot_out = sum(s[2] for s in summary)
    print(f"合计：源 {tot_src} 行 -> 干净 {tot_out} 条，剔除重复/虚胖 {tot_src - tot_out} 行 "
          f"({(tot_src-tot_out)/max(tot_src,1)*100:.1f}%)")


if __name__ == "__main__":
    main()

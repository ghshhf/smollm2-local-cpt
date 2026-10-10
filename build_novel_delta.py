"""生成「小说设定卡 · 补卡」数据集。

背景：V3 训练跑的是旧数据 role_data_v3.jsonl（624 行 / 唯一 575 个话题），
而清洗后的 card_novel.jsonl 有 630 个话题，是 V3 的**严格超集**（完全包含那 575 个，多出 55 个）。

所以没必要为了这 55 个话题把整张卡推倒重来（那样要把已跑的 320 步作废）。正确做法是：
  新话题（55 条）+ 旧话题回放（随机抽一批）-> 从 out_lora_v3 的权重**接着练**
  （replay 是防灾难性遗忘的标准配方：只喂新数据会让模型把旧设定忘掉）

用法：python build_novel_delta.py [--replay 130]
"""
import argparse
import json
import os
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(ROOT, "data", "role_data_v3.jsonl")
NOVEL = os.path.join(ROOT, "data", "card_novel.jsonl")
OUT = os.path.join(ROOT, "data", "card_novel_delta.jsonl")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, default=130, help="从 V3 已学过的旧话题里回抽多少条做防遗忘")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v3 = load(V3)
    novel = load(NOVEL)
    # V3 里已经见过的话题（用 instruction 判重，与 build_cards_by_domain 口径一致）
    seen = {r["instruction"] for r in v3}

    new = [r for r in novel if r["instruction"] not in seen]
    # 去重，避免新话题里自己抄自己
    uniq, memory = [], set()
    for r in new:
        if r["instruction"] in memory:
            continue
        memory.add(r["instruction"])
        uniq.append(r)

    # 回放池从 V3 的唯一话题里抽，按 instruction 去重
    seen2, old_pool = set(), []
    for r in v3:
        k = r["instruction"]
        if k in seen2:
            continue
        seen2.add(k)
        old_pool.append(r)

    random.seed(args.seed)
    k = min(args.replay, len(old_pool))
    replay = random.sample(old_pool, k)

    rows = uniq + replay
    random.shuffle(rows)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"补卡数据集已生成 -> {OUT}")
    print(f"  新话题(V3 没学过的): {len(uniq)}")
    print(f"  旧话题回放(防遗忘): {len(replay)} / 池 {len(old_pool)}")
    print(f"  合计: {len(rows)} 条  -> 约 {int(len(rows) * 1.6 / 4)} 步(4 样本/步, ~1.6 epoch)")


if __name__ == "__main__":
    main()

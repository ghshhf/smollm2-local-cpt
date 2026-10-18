# -*- coding: utf-8 -*-
"""synth_chat_data_v7.py — 北极星对话语料 v7 合成器（教师种子 + 组合扩展）。

输入：data/chat_seeds/seeds_v7.py / seeds_v7b.py / seeds_v7c.py
      每个 SEEDS = [(域, [问法...], [答法...]), ...]
处理：
  1) 展开：len(问法) x len(答法) 全组合（单轮）
  2) 多轮拼接：同域内抽样拼成 2~3 轮真实对话（日常/身份/能力/情感/主人偏好域）
  3) 去重、长度约束、域配额
输出：data/chat_sft_v7.jsonl 每行 {"text": ChatML}
       data/chat_sft_v7.stats.json 域分布

用法：
  python tools/synth_chat_data_v7.py --target 3000 --out data/chat_sft_v7.jsonl --seed 7
"""
import argparse
import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEED_DIR = os.path.join(ROOT, "data", "chat_seeds")

U, A = "<|im_start|>user\n", "<|im_start|>assistant\n"
E = "<|im_end|>"

# 多轮只在"能自然连着聊"的域做；knowledge/opinion 也可连问，但优先日常
MULTI_DOMAINS = ["greeting", "emotion", "ability", "owner", "identity", "opinion",
                 "knowledge", "clarify", "work", "study", "life", "tech"]
# 域配额权重（日常/身份主导，按 RESULT.md §6）
DOMAIN_WEIGHT = {
    "identity": 1.15, "greeting": 1.15, "emotion": 1.15, "owner": 1.10, "ability": 1.05,
    "opinion": 0.80, "knowledge": 0.80, "clarify": 0.85, "boundary": 0.85,
}


def load_seeds():
    out = []
    for fn in ("seeds_v7.py", "seeds_v7b.py", "seeds_v7c.py", "seeds_v7d.py"):
        p = os.path.join(SEED_DIR, fn)
        if not os.path.exists(p):
            print("[warn] 缺 %s" % p, flush=True)
            continue
        spec = importlib.util.spec_from_file_location("seedmod_" + fn[:-3], p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        out.extend(m.SEEDS)
        print("[seed] %-14s %d 组" % (fn, len(m.SEEDS)), flush=True)
    return out


def one_turn(q, a):
    return "%s%s%s\n%s%s%s" % (U, q, E, A, a, E)


def two_turn(q1, a1, q2, a2):
    return "%s%s%s\n%s%s%s\n%s%s%s\n%s%s%s" % (
        U, q1, E, A, a1, E, U, q2, E, A, a2, E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "chat_sft_v7.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-chars", type=int, default=520, help="单条最大字符数(约 1024 token 内)")
    ap.add_argument("--multi-ratio", type=float, default=0.34, help="多轮样本占比")
    ap.add_argument("--stats", default=os.path.join(ROOT, "data", "chat_sft_v7.stats.json"))
    a = ap.parse_args()
    rng = random.Random(a.seed)

    seeds = load_seeds()
    if not seeds:
        print("[fatal] 没有种子", flush=True)
        return 1

    # ---------- 1. 单轮展开（一问一答，索引配对，禁同问多答）----------
    by_domain = {}
    for dom, qs, as_ in seeds:
        for i, q in enumerate(qs):
            ans = as_[i % len(as_)]          # 一问一答：按索引轮转取答，绝不产生同问多答
            t = one_turn(q.strip(), ans.strip())
            if len(t) <= a.max_chars:
                by_domain.setdefault(dom, []).append(t)

    single_n = sum(len(v) for v in by_domain.values())
    print("[expand] 单轮一问一答 %d 条，域: %s" % (
        single_n, {k: len(v) for k, v in sorted(by_domain.items())}), flush=True)

    # ---------- 2. 多轮拼接（完整 prompt 唯一，同一对限曝光）----------
    multi = {d: [] for d in MULTI_DOMAINS}
    seen_round = set()
    pool = {d: [] for d in MULTI_DOMAINS}
    for dom, qs, as_ in seeds:
        if dom in pool:
            for i, q in enumerate(qs):
                pool[dom].append((q.strip(), as_[i % len(as_)].strip()))

    pair_uses = {}
    import collections as _col
    pair_uses = _col.Counter()
    PAIR_CAP = 30                                    # 同一 (问,答) 对在多轮里最多出现次数

    for _ in range(a.target * 8):
        d = rng.choice(MULTI_DOMAINS)
        if len(pool[d]) < 2:
            continue
        (q1, a1) = rng.choice(pool[d])
        (q2, a2) = rng.choice(pool[d])
        if q1 == q2:
            continue
        if pair_uses[(q1, a1)] >= PAIR_CAP or pair_uses[(q2, a2)] >= PAIR_CAP:
            continue
        t = two_turn(q1, a1, q2, a2)
        if len(t) > a.max_chars or t in seen_round:
            continue
        seen_round.add(t)
        pair_uses[(q1, a1)] += 1
        pair_uses[(q2, a2)] += 1
        multi[d].append(t)

    multi_n = sum(len(v) for v in multi.values())
    print("[multi] 多轮唯一候选 %d 条，域: %s" % (
        multi_n, {k: len(v) for k, v in sorted(multi.items())}), flush=True)

    # ---------- 3. 配额：单轮全用（唯一问法），余额由多轮补足 ----------
    def weighted_pick(items, k):
        """按域权重加权抽样（不放回，去重）。"""
        if not items or k <= 0:
            return []
        idxs, w = list(range(len(items))), [DOMAIN_WEIGHT.get(d, 1.0) for d, _ in items]
        picked, guard = [], 0
        while len(picked) < k and idxs and guard < k * 40:
            guard += 1
            j = rng.choices(idxs, weights=[w[i] for i in idxs], k=1)[0]
            picked.append(items[j])
            idxs.remove(j)
        return picked

    all_single, all_multi = [], []
    for d, lst in by_domain.items():
        for t in lst:
            all_single.append((d, t))
    for d, lst in multi.items():
        for t in lst:
            all_multi.append((d, t))

    rng.shuffle(all_single)
    rng.shuffle(all_multi)

    chosen, used = [], set()
    for d, t in all_single:                       # 单轮全覆盖
        if t in used:
            continue
        used.add(t)
        chosen.append((d, t))

    need = max(0, a.target - len(chosen))
    add_multi = weighted_pick(all_multi, need)    # 余额按域权重从多轮取
    for d, t in add_multi:
        if t in used:
            continue
        used.add(t)
        chosen.append((d, t))

    # 仍不足则从剩余多轮硬补
    if len(chosen) < a.target:
        rest = [(d, t) for d, t in all_multi if t not in used]
        rng.shuffle(rest)
        for d, t in rest[: a.target - len(chosen)]:
            used.add(t)
            chosen.append((d, t))

    chosen = chosen[: a.target]

    # 最终全局去重（防单轮/多轮拼出同一条）
    _seen, _uniq = set(), []
    for d, t in chosen:
        if t in _seen:
            continue
        _seen.add(t)
        _uniq.append((d, t))
    chosen = _uniq

    # 去重后不足则从剩余多轮补足
    if len(chosen) < a.target:
        used2 = set(t for _, t in chosen)
        rest = [(d, t) for d, t in all_multi if t not in used2]
        rng.shuffle(rest)
        chosen += rest[: a.target - len(chosen)]
    chosen = chosen[: a.target]

    # ---------- 4. 写盘 ----------
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for _, t in chosen:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    dist = {}
    for d, _ in chosen:
        dist[d] = dist.get(d, 0) + 1
    stats = {
        "total": len(chosen),
        "unique": len(set(t for _, t in chosen)),
        "single_pool": single_n,
        "multi_pool": multi_n,
        "domain_dist": dict(sorted(dist.items(), key=lambda x: -x[1])),
        "avg_chars": round(sum(len(t) for _, t in chosen) / max(1, len(chosen)), 1),
        "out": a.out,
    }
    with open(a.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n[out] %s" % a.out, flush=True)
    print("[stats] total=%d unique=%d avg_chars=%s" % (
        stats["total"], stats["unique"], stats["avg_chars"]), flush=True)
    for d, n in stats["domain_dist"].items():
        print("   %-10s %4d" % (d, n), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

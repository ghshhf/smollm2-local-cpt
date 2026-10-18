"""合并编程+推理指令卡为 0.2B 续训语料（corpus_code_v1）。

原则（沿用本项目纪律）：
- 只并「父卡」不并子卡：card_code.jsonl(1463) 是父卡，4 张 card_code_* 子卡是它的拆分，
  并子卡会造成重复，故只取父卡。
- card_reason.jsonl(1471) 一并并入（推理与编程同属「结构化输出」能力，用户三线规划里
  编程线含推理底子）。
- 去重口径与 build_cards_by_domain 一致：**(instruction, output) 精确对去重**。
  注意：card_code 与 card_reason 有 1390 条同题不同答（推理卡是代码题的 CoT 展开），
  按 instruction 去重会整段丢掉推理链 ⇒ 必须按题+答去重，两种答法都留。
- 格式归一为 {"instruction":..., "output":...}，train_smollm2_pt.py 原生兼容。
- 产出必须过 tools/inspect_jsonl.py 体检后才算数。

用法：python tools/build_code_corpus.py
输出：data/corpus_code_v1.jsonl + 统计打印
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = [
    os.path.join(ROOT, "data", "card_code.jsonl"),    # 父卡优先
    os.path.join(ROOT, "data", "card_reason.jsonl"),
]
OUT = os.path.join(ROOT, "data", "corpus_code_v1.jsonl")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                print("  [bad] %s:%d" % (os.path.basename(path), ln))
                continue
            ins = (r.get("instruction") or "").strip()
            out = (r.get("output") or "").strip()
            if not ins or not out:
                continue
            rows.append({"instruction": ins, "output": out})
    return rows


def main():
    seen = {}
    order = []
    total_in = 0
    for p in SRC:
        rows = load(p)
        total_in += len(rows)
        for r in rows:
            k = (r["instruction"], r["output"])
            if k in seen:
                continue
            seen[k] = r
            order.append(k)
        print("[load] %-24s %5d 条" % (os.path.basename(p), len(rows)))
    merged = [seen[k] for k in order]
    with open(OUT, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_tok = sum(len(r["instruction"]) + len(r["output"]) for r in merged) // 2  # 粗估 token（中文≈1.5字符/token 的折中）
    print("[dedup] 入 %d 条 -> 出精确去重后 %d 条（instruction+output 维度；1390 条同题双答保留）" % (total_in, len(merged)))
    print("[out ] %s  (%.1f MB, 粗估 %.1fM token)" % (OUT, os.path.getsize(OUT) / 1e6, n_tok / 1e6))
    # 最短/最长 output 分布，防超短样本稀释
    ls = sorted(len(r["output"]) for r in merged)
    print("[len ] output 字符数 p10=%d p50=%d p90=%d max=%d" % (
        ls[int(len(ls) * 0.1)], ls[len(ls) // 2], ls[int(len(ls) * 0.9)], ls[-1]))


if __name__ == "__main__":
    main()

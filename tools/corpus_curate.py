#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""corpus_curate.py — 代码语料提纯 + 合并 dream 燃料（代码模块 v2 用）

背景（2026-10-18 代码 LoRA v1 负迁移裁决）：card_code.jsonl 1463 条里
22%(323条)是「纯函数定义、无顶层调用」、72% 不含 print——1 epoch 就把
Qwen「写完整可执行脚本」的习惯冲成「写定义不调用」，strict 反比裸模低。
本脚本把这类样本清掉，只保留「定义+调用+期望输出」三位一体的语料，
再合并 node1 做梦银行产出的 dream_fuel.jsonl（天然可执行+产视频形态）。

用法:
  python tools/corpus_curate.py            # 读 card_code.jsonl [+dream_fuel.jsonl] → data/code_sft_v2.jsonl
  python tools/corpus_curate.py --dry-run  # 只统计不写盘
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CARD = os.path.join(ROOT, "data", "card_code.jsonl")
FUEL = os.path.join(ROOT, "data", "dream_fuel.jsonl")
OUT = os.path.join(ROOT, "data", "code_sft_v2.jsonl")

# 顶层可执行语句（非 def/class/import/#/docstring 行内的实质动作）
_TOP_EXEC = re.compile(
    r"^(?!\s*(?:def |class |import |from |#|\"\"\"|'''|@))\s*"
    r"(?:print\(|= .*\(|assert |for |while |if __|if __name__|return |[A-Za-z_]\w*\(.*\)\s*$)",
    re.M)


def extract_code(output):
    m = re.search(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", output or "", re.S)
    return m.group(2) if m else (output or "")


def is_executable(code):
    """有顶层可执行语句才算「完整脚本」；纯 def/class 集合=半成品。"""
    return bool(_TOP_EXEC.search(code))


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    seen, kept, dropped = set(), [], {"pure_def": 0, "dup": 0, "empty": 0}

    for r in load_jsonl(CARD):
        code = extract_code(r.get("output", "")).strip()
        inst = (r.get("instruction") or "").strip()
        key = (inst, code)
        if not code or not inst:
            dropped["empty"] += 1
            continue
        if key in seen:
            dropped["dup"] += 1
            continue
        seen.add(key)
        if not is_executable(code):
            dropped["pure_def"] += 1
            continue
        kept.append({"instruction": inst,
                     "output": "```python\n%s\n```\n" % code,
                     "_type": "code", "_universe": "代码",
                     "_src": "card_code_curated"})

    n_card = len(kept)
    n_fuel = 0
    for r in load_jsonl(FUEL):
        code = extract_code(r.get("output", "")).strip()
        inst = (r.get("instruction") or "").strip()
        if not code or not inst or (inst, code) in seen:
            continue
        seen.add((inst, code))
        # dream 燃料天然是「可执行+产视频」形态，不再过滤
        kept.append({"instruction": inst,
                     "output": "```python\n%s\n```\n" % code,
                     "_type": "code_rsi", "_universe": "代码",
                     "_src": "dream_triple_bank"})
        n_fuel += 1

    print("[curate] card_code 提纯后=%d  (dropped: %s)" % (n_card, dropped))
    print("[curate] + dream_fuel=%d  => total=%d" % (n_fuel, len(kept)))
    if a.dry_run:
        return
    with open(OUT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[curate] -> %s" % OUT)


if __name__ == "__main__":
    main()

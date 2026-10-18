#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_triples_to_sft.py — 做梦自进化产出 → polaris 代码 SFT 训练燃料（RSI 收口 #3）

把 triple_bank.db 里「过了 vid_verify + task_verify 双闸门」的 (task, code_text)
导出成与 card_code.jsonl 完全兼容的格式，供晚上 GPU 回训 polaris 编程模块
（LoRA 只挂 q/k/v/o）使用。

为什么能当燃料：triples 里的代码是 Qwen-0.5B 做梦者的合格产物（视频能播 + 要素齐备），
polaris 代码底座本就为 0，用强模型产物蒸馏其代码能力，正是 RSI 自举的意图——
模型变强 → 做梦产出更好 → 回训更强。

格式（对齐 data/card_code.jsonl）：
  {"instruction": <task>, "output": "```python\n<code>\n```\n",
   "_type":"code_rsi", "_universe":"代码", "_src":"dream_triple_bank",
   "_triple_id":<id>, "_created_at":<ts>}

用法:
  python tools/export_triples_to_sft.py                # 默认读 data/triple_bank.db → data/dream_code_sft.jsonl
  python tools/export_triples_to_sft.py --min-quality 0.8
  python tools/export_triples_to_sft.py --limit 50
"""
import argparse
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "data", "triple_bank.db")
OUT = os.path.join(ROOT, "data", "dream_code_sft.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB, help="triple_bank.db 路径")
    ap.add_argument("--out", default=OUT, help="输出 jsonl 路径")
    ap.add_argument("--min-quality", type=float, default=0.0,
                    help="按 checks_json 的 quality 过滤(0=不过滤)")
    ap.add_argument("--limit", type=int, default=0, help="最多导出条数(0=全部)")
    a = ap.parse_args()

    if not os.path.isfile(a.db):
        print("ERR db 不存在: %s" % a.db)
        sys.exit(2)

    c = sqlite3.connect(a.db)
    rows = c.execute(
        "SELECT id,task,code_text,checks_json,created_at FROM triples ORDER BY id"
    ).fetchall()

    seen, n, skip = set(), 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for id_, task, code, checks_json, created_at in rows:
            if a.limit and n >= a.limit:
                break
            code = (code or "").strip()
            if not code:
                skip += 1
                continue
            if code in seen:          # 按代码内容去重，避免重复样本
                skip += 1
                continue
            seen.add(code)
            if a.min_quality > 0 and checks_json:
                try:
                    q = float(json.loads(checks_json).get("quality", 0))
                    if q < a.min_quality:
                        skip += 1
                        continue
                except Exception:
                    pass
            rec = {
                "instruction": task,
                "output": "```python\n%s\n```\n" % code,
                "_type": "code_rsi",
                "_universe": "代码",
                "_src": "dream_triple_bank",
                "_triple_id": id_,
                "_created_at": created_at,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print("OK 导出 %d 条(跳过 %d 条重复/空/低质) → %s" % (n, skip, a.out))
    print("   合并进 data/card_code.jsonl 即可回训 polaris 编程模块(LoRA q/k/v/o)。")


if __name__ == "__main__":
    main()

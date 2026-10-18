# -*- coding: utf-8 -*-
"""
chat_sft_v2 修正版（2026-10-17 18:20 本会话接手）
- 保留: 现有 v2 中非模板腔的条目（北极星身份层 + 基础对话 + 真代码卡）
- 剔除: 80 条 "思考步骤（启发式分析）" 机器人腔模板（card_reason 采样, 对对话目标是噪声）
- 合并: 素材池 100 条日常对话（build_chat_data_v2.D 手写）, 身份口径统一为北极星
- 备份: 原文件 -> chat_sft_v2_raw.jsonl
"""
import json
import re
import shutil
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from build_chat_data_v2 import D  # 手写日常对话 (q, a) 列表

P = r"D:\work\_local_ai\train_15b\data\chat_sft_v2.jsonl"
RAW = r"D:\work\_local_ai\train_15b\data\chat_sft_v2_raw.jsonl"
POOL_OUT = r"D:\work\_local_ai\train_15b\data\chat_extras_pool.jsonl"


def q_of(text):
    m = re.search(r"user\n(.*?)<\|im_end\|>", text)
    return (m.group(1).strip() if m else "")


def wrap(q, a):
    return {"text": "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s<|im_end|>" % (q, a)}


def main():
    shutil.copy(P, RAW)  # 备份

    rows = [json.loads(l) for l in open(P, encoding="utf-8")]
    kept, dropped_tpl = [], 0
    for r in rows:
        if "思考步骤" in r["text"]:
            dropped_tpl += 1
            continue
        kept.append(r)
    print("[v2原] %d 条, 剔模板腔 %d, 保留 %d" % (len(rows), dropped_tpl, len(kept)))

    seen = {q_of(r["text"]) for r in kept}

    # 素材池: 身份口径统一北极星; 与 v2 撞题的让位（v2 的北极星版优先）
    FIX = {
        "你有名字吗": "有，我叫北极星，主人给定的。随叫随应。",
    }
    SKIP_IF_IN_V2 = {"你叫什么名字", "你是谁", "你会什么", "北极星", "主人"}  # v2 北极星版优先
    added = 0
    for q, a in D:
        q = q.strip()
        a = FIX.get(q, a)
        if q in SKIP_IF_IN_V2 and q in seen:
            continue
        if q in seen:
            continue
        seen.add(q)
        kept.append(wrap(q, a))
        added += 1
    print("[素材池] 新增 %d 条（身份口径=北极星）" % added)

    with open(P, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[v2修正] 共 %d 条 -> %s (原版备份 %s)" % (len(kept), P, RAW))

    # 同步刷新素材池文件（含修正后口径）
    with open(POOL_OUT, "w", encoding="utf-8") as f:
        for q, a in D:
            f.write(json.dumps(wrap(q, FIX.get(q.strip(), a)), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

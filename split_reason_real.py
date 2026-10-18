# -*- coding: utf-8 -*-
"""去虚胖：推理卡只留真 CoT，启发式占位隔离到 quarantine，不进训练。

输出:
  data/card_reason_real.jsonl          123 条真推理链(可训练)
  data/card_reason_quarantine.jsonl    1911 条占位(冻结,待老师回填)
"""
import json, os, io, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.abspath(__file__))

SRC = os.path.join(ROOT, "data", "role_data_v7_selftaught.jsonl")
REAL = os.path.join(ROOT, "data", "card_reason_real.jsonl")
QUAR = os.path.join(ROOT, "data", "card_reason_quarantine.jsonl")

def is_placeholder(r):
    """占位判定：显式标记，或思考段以模板句式开头。"""
    blob = str(r.get("output", "")) + str(r.get("instruction", ""))
    if "启发式" in blob or "非 LLM 推理" in blob:
        return True
    out = str(r.get("output", ""))
    # 占位思考段的常见开头
    for head in ("思考步骤（", "1. 任务：", "2. 语言/结构：", "3. 关键边界："):
        if out.strip().startswith(head):
            return True
    return False

n_real = n_quar = 0
with open(SRC, encoding="utf-8") as fi, \
     open(REAL, "w", encoding="utf-8") as fr, \
     open(QUAR, "w", encoding="utf-8") as fq:
    for line in fi:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if is_placeholder(r):
            fq.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_quar += 1
        else:
            fr.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_real += 1

print(f"真 CoT  -> card_reason_real.jsonl       {n_real} 条")
print(f"占位    -> card_reason_quarantine.jsonl {n_quar} 条（冻结，不进训练）")

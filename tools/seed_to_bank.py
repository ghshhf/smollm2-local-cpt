# -*- coding: utf-8 -*-
"""seed_to_bank.py — 把 data/seed_code_200.jsonl 灌进 triple_bank（代码线底层数据）

目的：做梦/自进化环启动前必须有底层粮。100 条手写种子先落库，
让 dream_code 的 bank_load 能读到初始 ngrams，去重闸门与导出才转得起来。

入库规则（与 dream_code.bank_add 完全一致）：
  - code_text 存纯 body（不含 ``` 围栏），方便 export_triples_to_sft 直接包块
  - video_path=None（代码线无视频）
  - checks_json 存 {seed, prose, src}，prose 是代码块外的说明
  - ngrams 用 triple_bank.ngrams，与做梦器去重同口径
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from triple_bank import conn, ngrams  # noqa

BLOCK_RE = re.compile(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", re.S)
IM_END = "<|im_end|>"


def main():
    c = conn()
    src = os.path.join(ROOT, "data", "seed_code_200.jsonl")
    n_add = n_skip = n_dup = 0
    for line in open(src, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        ins = (o.get("instruction") or "").strip()
        out = o.get("output") or ""
        m = BLOCK_RE.search(out)
        if not m or m.group(1).lower() not in ("", "python"):
            n_skip += 1
            continue
        body = m.group(2).strip("\n")
        if not body.strip():
            n_skip += 1
            continue
        # 块外说明
        outside = re.sub(r"```[A-Za-z0-9_+-]*\s*\n.*?```", "", out, flags=re.S)
        outside = outside.replace(IM_END, "").strip()
        prose = outside[:300]
        dup = c.execute("SELECT id FROM triples WHERE task=?", (ins,)).fetchone()
        if dup:
            n_dup += 1
            continue
        checks = json.dumps({"seed": True, "prose": prose, "src": "seed_code_200"},
                            ensure_ascii=False)
        c.execute("""INSERT INTO triples(task,code_path,code_text,video_path,checks_json,ngrams)
                     VALUES(?,?,?,?,?,?)""",
                  (ins, None, body, None, checks,
                   json.dumps(ngrams(body), ensure_ascii=False)))
        n_add += 1
    c.commit()
    n = c.execute("SELECT count(*) FROM triples").fetchone()[0]
    print(f"OK 灌入 {n_add} 条, 跳过无块 {n_skip} 条, 重复 {n_dup} 条, 库内共 {n} 条")
    c.close()


if __name__ == "__main__":
    main()

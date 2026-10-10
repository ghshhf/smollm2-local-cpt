#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性诊断：扫描 WorkBuddy 会话存档里所有 user 轮的首标签分布，
用于收齐平台注入块的完整清单（供 harvest 脚本拉黑）。"""
import json, os, re
from collections import Counter

root = os.environ.get("WB_PROJECTS", "C:/Users/admin/.workbuddy/projects")
RE_START = re.compile(r"^\s*<([a-zA-Z_][\w-]*)")
cnt = Counter()
tot = 0

for dp, _, files in os.walk(root):
    for fn in files:
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(dp, fn), encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                objs = o if isinstance(o, list) else [o]
                for x in objs:
                    role = (x.get("role") or (x.get("message") or {}).get("role") or "").lower()
                    if role not in ("user", "human"):
                        continue
                    c = x.get("content") or (x.get("message") or {}).get("content") or x.get("parts")
                    if isinstance(c, list):
                        c = "\n".join((p.get("text", "") if isinstance(p, dict) else str(p)) for p in c)
                    if not isinstance(c, str):
                        continue
                    tot += 1
                    m = RE_START.match(c)
                    if m:
                        cnt[m.group(1).lower()] += 1

print("user 轮总数:", tot)
print("=== 首标签分布 ===")
for k, v in cnt.most_common(60):
    print(f"  <{k}> : {v}")

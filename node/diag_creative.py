#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_creative.py — 诊断 l3_creative_fill 现场状态并回传 ECS。"""
import os, subprocess, time

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
D = "/mnt/workspace/node1"
lines = []
def L(m):
    lines.append(str(m)); print(m, flush=True)

L("=== diag time %s ===" % time.strftime("%Y-%m-%d %H:%M:%S"))
l3 = os.path.join(D, "corpus_04b", "l3_ms")
if os.path.isdir(l3):
    for f in sorted(os.listdir(l3)):
        p = os.path.join(l3, f)
        L("l3_ms/%-28s %.1fMB" % (f, os.path.getsize(p)/1e6))
else:
    L("l3_ms MISSING")
raw = os.path.join(l3, "l3_multistyle.jsonl")
if os.path.exists(raw):
    L("l3_multistyle.jsonl %.1fMB" % (os.path.getsize(raw)/1e6))
else:
    L("l3_multistyle.jsonl MISSING")
cr = os.path.join(D, "corpus_04b", "creative_refined")
if os.path.isdir(cr):
    for f in sorted(os.listdir(cr)):
        L("creative_refined/%-28s %.1fMB" % (f, os.path.getsize(os.path.join(cr, f))/1e6))
else:
    L("creative_refined MISSING")
log = os.path.join(D, "creative_fill.log")
if os.path.exists(log):
    L("=== creative_fill.log tail ===")
    with open(log, encoding="utf-8", errors="replace") as fh:
        L("".join(fh.readlines()[-25:]))
else:
    L("creative_fill.log MISSING")
try:
    out = subprocess.run(["pgrep", "-af", "l3_creative_fill"], capture_output=True, text=True, timeout=10).stdout
    L("=== pgrep l3_creative_fill ===\n" + (out or "NONE RUNNING"))
except Exception as e:
    L("pgrep err %s" % e)
rep = "\n".join(lines).encode("utf-8")
try:
    subprocess.run(["curl", "-s", "-X", "POST",
                    "%s/upload?name=diag_creative.txt&key=%s" % (ECS, KEY),
                    "--data-binary", rep], timeout=20)
    L("DIAG uploaded")
except Exception as e:
    L("DIAG upload fail %s" % e)

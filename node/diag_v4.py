#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag v4: node1 本地磁盘余量 + 旧任务 CPU 时间 + cand/jsonl 大小"""
import os, subprocess, time

ECS = "http://47.99.131.12:8081"; KEY = "k0u4mail"; TS = time.strftime("%H%M%S")
D = "/mnt/workspace/node1/corpus_04b"

lines = []
def L(x): lines.append(str(x)); print(x, flush=True)

L("===== diag_v4 @ %s =====" % TS)
# 磁盘
out = subprocess.run(["bash","-c","df -h /mnt/workspace/node1 /tmp / 2>/dev/null; echo ---; df -h | head -n 20"],
                     capture_output=True, text=True, timeout=20).stdout
L("[disk]\n"+out)
# 旧任务 CPU 时间
ps = subprocess.run(["bash","-c","ps -o pid,etime,time,pcpu,rss,cmd -p $(pgrep -f l3_creative_fill.py | head -1) 2>/dev/null"],
                   capture_output=True, text=True, timeout=20).stdout
L("[proc]\n"+ (ps.strip() or "（无进程）"))
# 大小
for f in ["l3_ms/l3_multistyle.jsonl", "creative_refined/.creative.cand", "creative_refined/creative_articles_l3.jsonl"]:
    p = os.path.join(D, f)
    if os.path.exists(p): L("[size] %s = %.1fMB" % (f, os.path.getsize(p)/1e6))
    else: L("[size] %s = 不存在" % f)

report = "\n".join(lines)
subprocess.run(["curl","-s","-X","POST","%s/upload?name=diag_v4_%s.txt&key=%s"%(ECS,TS,KEY),
                "--data-binary",report.encode("utf-8")],timeout=20)
subprocess.run(["curl","-s","-X","POST","%s/upload?name=diag_v4_FINAL.txt&key=%s"%(ECS,KEY),
                "--data-binary",report.encode("utf-8")],timeout=20)
L("[upload] diag_v4 done")

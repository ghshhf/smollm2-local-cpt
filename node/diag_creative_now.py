#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 0.4B 创意文章 L3 精炼任务在 node1 的真实状态"""
import os, subprocess, time, glob

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
TS = time.strftime("%Y%m%d_%H%M%S")
D = "/mnt/workspace/node1/corpus_04b"
OUT = os.path.join(D, "creative_refined")
L3 = os.path.join(D, "l3_ms")

lines = []
def L(x):
    lines.append(str(x)); print(x, flush=True)

L("===== diag_creative @ %s =====" % TS)
# 1) 进程
try:
    out = subprocess.run(["bash", "-c", "ps -ef | grep l3_creative_fill | grep -v grep"],
                         capture_output=True, text=True, timeout=20).stdout
    L("[proc] l3_creative_fill 进程:\n%s" % (out.strip() or "（无，已退出）"))
except Exception as e:
    L("[proc] err %s" % e)

# 2) 输出目录
def du(path):
    try:
        return subprocess.run(["bash","-c","ls -la '%s' 2>/dev/null | tail -n +2" % path],
                              capture_output=True, text=True, timeout=20).stdout
    except Exception as e:
        return "err %s"%e
L("[out] creative_refined/:\n%s" % (du(OUT).strip() or "（空）"))
L("[l3] l3_ms/:\n%s" % (du(L3).strip() or "（空）"))

# 3) 关键文件大小
for f in ["creative_articles_l3.jsonl", ".creative.cand"]:
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        L("[file] %s = %.1fMB" % (f, os.path.getsize(p)/1e6))
    else:
        L("[file] %s = 不存在" % f)

# 4) 是否有 log
log = os.path.join(D, "creative_fill.log")
if os.path.exists(log):
    L("[log] creative_fill.log 尾部:\n" + subprocess.run(["tail","-n","25",log],
        capture_output=True,text=True,timeout=20).stdout)

# 回传
report = "\n".join(lines)
try:
    subprocess.run(["curl","-s","-X","POST","%s/upload?name=diag_creative_%s.txt&key=%s"%(ECS,TS,KEY),
                    "--data-binary",report.encode("utf-8")],timeout=20)
    L("[upload] -> diag_creative_%s.txt ok" % TS)
    # 固定名双传，避免本地不知 TS 无法取回
    subprocess.run(["curl","-s","-X","POST","%s/upload?name=diag_creative_FINAL.txt&key=%s"%(ECS,KEY),
                    "--data-binary",report.encode("utf-8")],timeout=20)
    L("[upload] -> diag_creative_FINAL.txt ok")
except Exception as e:
    L("[upload] fail %s"%e)

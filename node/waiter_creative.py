#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waiter_creative.py — node1 后台看门狗：等 l3_creative_fill 产出 creative_articles_l3.jsonl，
稳定后读 creative_fill.log 汇总 + wc -l，回传完成报告。
跑法: (nohup python3 waiter_creative.py > waiter.log 2>&1 &)
"""
import os, subprocess, time

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
OUT = "/mnt/workspace/node1/corpus_04b/creative_refined/creative_articles_l3.jsonl"
CAND = "/mnt/workspace/node1/corpus_04b/creative_refined/.creative.cand"
LOG = "/mnt/workspace/node1/creative_fill.log"
DONE_NAME = "creative_fill_DONE.txt"
HB_NAME = "creative_fill_HB.txt"
TIMEOUT = 45 * 60

def post(name, text):
    try:
        subprocess.run(["curl", "-s", "-X", "POST",
                        "%s/upload?name=%s&key=%s" % (ECS, name, KEY),
                        "--data-binary", text.encode("utf-8")], timeout=25)
    except Exception as e:
        pass

def proc_count():
    try:
        return subprocess.run(["bash", "-c", "pgrep -f l3_creative_fill.py | grep -v $$ | wc -l"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return "?"

start = time.time()
last_size = -1
stable = 0
while time.time() - start < TIMEOUT:
    elapsed = int(time.time() - start)
    hb = ["waiter @%dmin" % (elapsed // 60)]
    if os.path.exists(OUT):
        sz = os.path.getsize(OUT)
        hb.append("OUTPUT_EXISTS %.1fMB" % (sz / 1e6))
        if sz == last_size:
            stable += 1
        else:
            stable = 0
            last_size = sz
        if stable >= 2 and sz > 1_000_000:
            # 输出稳定 -> 收口
            try:
                wc = subprocess.run(["wc", "-l", OUT], capture_output=True, text=True, timeout=30).stdout.split()[0]
            except Exception:
                wc = "?"
            logtail = ""
            if os.path.exists(LOG):
                logtail = subprocess.run(["tail", "-n", "22", LOG], capture_output=True, text=True, timeout=30).stdout
            report = ("CREATIVE_FILL_DONE\noutput=%s\n  size=%.1fMB  lines=%s\n  proc_running=%s\n"
                      "--- creative_fill.log 尾部 ---\n%s") % (OUT, sz / 1e6, wc, proc_count(), logtail)
            post(DONE_NAME, report)
            post(HB_NAME, report)
            break
    else:
        hb.append("OUTPUT_NOT_YET")
        if os.path.exists(CAND):
            hb.append("cand=%.1fMB" % (os.path.getsize(CAND) / 1e6))
        hb.append("proc=%s" % proc_count())
    post(HB_NAME, "\n".join(hb))
    time.sleep(60)
else:
    report = ("CREATIVE_FILL_TIMEOUT\n45min 内未产出稳定 output。旧精炼任务可能异常退出，"
             "需在本机修复脚本后重派（参考 l3_creative_fill.py 的 report NameError 修复）。")
    post(DONE_NAME, report)
    post(HB_NAME, report)

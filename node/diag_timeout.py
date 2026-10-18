# -*- coding: utf-8 -*-
"""诊断：L3 创意精炼任务是否还活着 / 死在哪 / 产物有没有"""
import os, subprocess, time

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
TS = time.strftime("%H%M%S")
lines = []

def L(x):
    lines.append(str(x)); print(x, flush=True)

def sh(cmd, t=30):
    try:
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=t).stdout
    except Exception as e:
        return "ERR:%s" % e

L("=== diag timeout @ %s ===" % TS)

L("[proc l3_creative / waiter]\n" + sh(
    "ps -eo pid,etime,time,rss,cmd | grep -E 'l3_creative_fill|waiter_creative' | grep -v grep || echo NONE"))

L("[ps aux mem]\n" + sh("free -m 2>/dev/null | head -3"))

OUT = "/mnt/workspace/node1/corpus_04b/creative_refined"
L("[ls creative_refined]\n" + sh("ls -la %s 2>/dev/null || echo NO_DIR" % OUT))

L("[find creative artifacts]\n" + sh(
    "find /mnt/workspace/node1 -maxdepth 3 \\( -name 'creative*' -o -name '*.cand' -o -name '*.jsonl' \\) "
    "-printf '%s\\t%p\\n' 2>/dev/null | sort -rn | head -25"))

log = "/mnt/workspace/node1/creative_fill.log"
if os.path.exists(log):
    L("[log size] %d bytes" % os.path.getsize(log))
    L("[log tail -n 45]\n" + sh("tail -n 45 " + log))
    L("[log head -n 12]\n" + sh("head -n 12 " + log))
else:
    L("[log] MISSING")

wl = "/mnt/workspace/node1/waiter.log"
if os.path.exists(wl):
    L("[waiter.log tail -n 15]\n" + sh("tail -n 15 " + wl))

L("[df]\n" + sh("df -h /mnt/workspace /tmp / 2>/dev/null"))

L("[all .log / .txt in node1]\n" + sh(
    "ls -la /mnt/workspace/node1/*.log /mnt/workspace/node1/*.txt 2>/dev/null | tail -20"))

rep = "\n".join(lines)
for nm in ("diag_timeout_%s.txt" % TS, "diag_timeout_FINAL.txt"):
    subprocess.run(["curl", "-s", "-X", "POST", "%s/upload?name=%s&key=%s" % (ECS, nm, KEY),
                    "--data-binary", rep.encode("utf-8")], timeout=25)

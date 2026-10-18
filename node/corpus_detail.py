#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""corpus_detail.py — 精查 R7/R8-R10 档位语料与 0.4B 语料实况。"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
CH = "/mnt/workspace/node1/chain"
ND = "/mnt/workspace/node1"

PARTS = [
    "echo '=== corpus_R7 实况 ==='; du -sh " + CH + "/corpus_R7/* 2>/dev/null; wc -l " + CH + "/corpus_R7/*.jsonl 2>/dev/null",
    "echo '--- R7 首行样品 ---'; head -c 260 " + CH + "/corpus_R7/zh_text_R7.jsonl 2>/dev/null",
    "echo ''; echo '=== R7 内文件清单(是否含四类) ==='; ls -la " + CH + "/corpus_R7/ 2>/dev/null | head -12",
    "echo '=== corpus_04b(0.4B) ==='; du -sh " + ND + "/corpus_04b 2>/dev/null; ls -la " + ND + "/corpus_04b/ 2>/dev/null | head -10",
    "echo '--- 04b 行数/首行 ---'; wc -l " + ND + "/corpus_04b/*.jsonl 2>/dev/null | head -5; head -c 200 " + ND + "/corpus_04b/*creative*novel*.jsonl 2>/dev/null",
    "echo ''; echo '=== 储备语料 refined ==='; ls -la " + CH + "/corpus_reserve_refined/ 2>/dev/null | head -10",
    "echo '=== 是否有 R8/R9/R10 档 ==='; ls -d " + CH + "/corpus_R8 " + CH + "/corpus_R9 " + CH + "/corpus_R10 2>/dev/null || echo 'R8/R9/R10 均不存在'",
    "echo '=== chain_v2.sh 课程定义 ==='; sed -n '178,215p' " + CH + "/chain_v2.sh 2>/dev/null",
]
CMD = ("{ " + "; ".join(PARTS) + "; } 2>&1 | curl -s -m90 -X POST '"
       + ECS + "/upload?name=corpus_detail.txt&key=" + KEY + "' --data-binary @-")

body = json.dumps({"cmd": CMD, "id": "cdet-%d" % int(time.time()), "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
urllib.request.urlopen(req, timeout=20).read()

txt = ""
for _ in range(24):
    time.sleep(5)
    try:
        txt = urllib.request.urlopen(
            "%s/file/corpus_detail.txt?key=%s" % (ECS, KEY), timeout=40
        ).read().decode("utf-8", "replace")
    except Exception:
        txt = ""
    if "课程定义" in txt:
        break
print(txt if txt.strip() else "(无回传)")

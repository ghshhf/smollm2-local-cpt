#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inventory node1 NAS corpus stock + R6 stage + disk, via ECS /task."""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
CMD = (
    "W=/mnt/workspace/node1/chain; "
    "{ echo '=== date ==='; date; "
    "echo '--- done_steps ---'; cat \"$W/done_steps.txt\" 2>/dev/null; "
    "echo '--- heartbeat ---'; cat \"$W/heartbeat\" 2>/dev/null; "
    "echo '--- stage/log tail ---'; tail -6 \"$W/chain.log\" 2>/dev/null; "
    "echo '--- disk ---'; df -h /mnt/workspace | tail -1; "
    "echo '--- node1 top dirs ---'; du -sh /mnt/workspace/node1/* 2>/dev/null | sort -rh | head -20; "
    "echo '--- corpus_reserve ---'; du -sh \"$W/corpus_reserve\"/* 2>/dev/null; "
    "echo '--- next_corpus_raw ---'; du -sh /mnt/workspace/node1/next_corpus_raw/* 2>/dev/null | head -20; "
    "echo '--- downloads dir ---'; ls -la /mnt/workspace/node1/downloads 2>/dev/null | head -20; "
    "} 2>&1 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=corpus_inv_001.txt&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "corpus-inv-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR:", e)

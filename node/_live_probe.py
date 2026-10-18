#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Send a fresh live-status probe to node1 via ECS /task; node1 uploads live_status_001.txt."""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
CMD = (
    "W=$(ls -d /mnt/workspace/node1/chain 2>/dev/null | head -1); "
    "{ echo '=== time ==='; date; "
    "echo \"W=$W\"; "
    "echo '--- done_steps ---'; cat \"$W/done_steps.txt\" 2>/dev/null; "
    "echo '--- heartbeat ---'; cat \"$W/heartbeat\" 2>/dev/null; "
    "echo '--- chain.pid ---'; P=$(cat \"$W/chain.pid\" 2>/dev/null); echo \"pid=$P\"; "
    "{ [ -n \"$P\" ] && kill -0 \"$P\" 2>/dev/null && echo 'chain ALIVE'; } || echo 'chain DEAD'; "
    "echo '--- trainer ---'; (ps -eo args 2>/dev/null | grep -v grep | grep -q 'train_remote' && echo 'trainer ALIVE') || echo 'trainer DEAD'; "
    "echo '--- last 10 train steps ---'; grep -E '\\[train\\] step|\\[val\\]' \"$W/chain.log\" 2>/dev/null | tail -10; "
    "echo '--- stage ---'; grep -oE 'stage=[A-Z0-9]+' \"$W/heartbeat\" 2>/dev/null; "
    "echo '--- log tail ---'; tail -15 \"$W/chain.log\" 2>/dev/null; "
    "echo '--- ckpt ---'; ls -la \"$W/ckpt/\" 2>/dev/null | tail -8; "
    "} 2>&1 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=live_status_001.txt&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "live-probe-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR:", e)

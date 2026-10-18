#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kill_build.py — 终止 node1 上挂死的 build_reserve(3GB 公众号 clone 卡住) + 清理半成品
结果上报 ECS /file/status_kill.txt
"""
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = (
    "pkill -9 -f build_reserve.py; pkill -9 -f 'git clone'; sleep 2; "
    "rm -rf /tmp/weixin_public_corpus; "
    "{ echo 'killed build_reserve+gitclone'; "
    "echo \"build_reserve_running=$(ps aux | grep -c '[b]uild_reserve.py')\"; "
    "echo '--- reserve dir ---'; ls -la /mnt/workspace/node1/chain/corpus_reserve; "
    "} | curl -s -X POST 'http://47.99.131.12:8081/upload?name=status_kill.txt&key=k0u4mail' --data-binary @-"
)
body = json.dumps({"cmd": CMD, "id": "kill-build", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:120])
except Exception as e:
    print("POST ERR:", e)

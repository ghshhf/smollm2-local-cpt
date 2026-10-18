#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""force_kill.py — 强杀 node1 上挂死的 git clone / build 进程(用 pkill -f git 匹配 git-remote-https)
解锁被 rm -rf 阻塞的 kill 任务, 并报告 node1 状态。
结果上报 ECS /file/status_forcekill.txt
"""
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = (
    "pkill -9 -f git; pkill -9 -f build_reserve; pkill -9 -f reserve_progress; sleep 3; "
    "rm -rf /tmp/weixin_public_corpus /tmp/reserve; "
    "{ echo 'force-killed git/build_reserve'; "
    "echo \"build_reserve_running=$(ps aux | grep -c '[b]uild_reserve.py')\"; "
    "echo \"git_running=$(ps aux | grep -c '[g]it ')\"; "
    "echo '--- reserve dir ---'; ls -la /mnt/workspace/node1/chain/corpus_reserve; "
    "} | curl -s -X POST 'http://47.99.131.12:8081/upload?name=status_forcekill.txt&key=k0u4mail' --data-binary @-"
)
body = json.dumps({"cmd": CMD, "id": "force-kill", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:120])
except Exception as e:
    print("POST ERR:", e)

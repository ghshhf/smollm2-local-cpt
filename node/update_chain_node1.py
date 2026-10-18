#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""update_chain_node1.py — 让 node1 把 ECS 上的新链(含 R6)原地覆盖到 $W/chain_v1.sh。
原地覆盖(同 inode): 正在跑的 sh 顺序读到 R5 循环之后即自然接到 R6 段, 不影响当前 R4/R5 训练。
结果上报到 ECS /file/status_chainupdate.txt
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
CMD = ("curl -s -m120 -o /mnt/workspace/node1/chain/chain_v1.sh http://47.99.131.12:8081/file/node1_chain.sh; "
       "{ echo UPDATED_LINES=$(wc -l < /mnt/workspace/node1/chain/chain_v1.sh); "
       "echo HASR6=$(grep -c 'R6 段' /mnt/workspace/node1/chain/chain_v1.sh); "
       "echo OLD_PID_RUNNING=$(ps aux|grep -c '[c]hain_v1.sh'); "
       "} | curl -s -X POST 'http://47.99.131.12:8081/upload?name=status_chainupdate.txt&key=k0u4mail' --data-binary @-")

body = json.dumps({"cmd": CMD, "id": "upd-chain", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR (task may still be queued):", e)

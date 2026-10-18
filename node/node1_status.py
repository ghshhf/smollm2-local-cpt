#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
node1_status.py — 查魔搭 node1 训练实时状态（自包含，零依赖，Windows/Linux 通用）。

原理：ECS 47.99.131.12:8081 是中继。本脚本 POST /task 派一条查询命令到 node1，
node1 执行后把结果 POST 回 /upload，本脚本再 GET /file 读回传并打印。

用法：python node1_status.py        （等约 10-30 秒出结果）
依赖：Python 3.8+ 标准库，无需 pip 装任何东西。
"""
import json
import time
import urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
NODE = "/mnt/workspace/node1"


def query_node1():
    tag = "sts_%d" % int(time.time())
    cmd = (
        "{ echo '=== TIME ==='; date '+%%Y-%%m-%%d %%H:%%M:%%S'; "
        "echo '=== DONE_STEPS(目标6800封顶后扩0.3B) ==='; cat %s/chain/done_steps.txt 2>/dev/null; "
        "echo '=== CHAIN_LOG_TAIL ==='; tail -5 %s/chain/chain.log 2>/dev/null; "
        "echo '=== TRAIN_PROCS ==='; pgrep -c -f train_remote_v4 2>/dev/null || echo 0; "
        "echo '=== HEARTBEAT ==='; cat %s/chain/heartbeat 2>/dev/null; "
        "echo '=== CKPT ==='; ls -t %s/chain/ckpt/ 2>/dev/null | head -3; } 2>&1 | "
        "curl -s -m60 -X POST '%s/upload?name=%s.txt&key=%s' --data-binary @-"
    ) % (NODE, NODE, NODE, NODE, ECS, tag, KEY)
    body = json.dumps({"cmd": cmd, "id": "status-" + tag, "ts": int(time.time())}).encode()
    req = urllib.request.Request(ECS + "/task", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=20).read()
    # node_v5 轮询器 10s 级响应；最多等 60s
    txt = ""
    for _ in range(15):
        time.sleep(4)
        try:
            txt = urllib.request.urlopen(
                "%s/file/%s.txt?key=%s" % (ECS, tag, KEY), timeout=30
            ).read().decode("utf-8", "replace")
        except Exception:
            txt = ""
        if "=== CHAIN_LOG_TAIL ===" in txt:
            break
    return txt


if __name__ == "__main__":
    out = query_node1()
    print(out if out.strip() else "(无回传：node1 可能正在回收重启，稍后重试)")

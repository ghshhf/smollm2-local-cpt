#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_health.py — node1 全面体检: 进程/权重完整性/日志推进/语料/停机标志/磁盘/0.3B链路就绪。
结果写 health<N>.txt (文件名轮换, 避 ECS 文件锁)。
"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
W = "/mnt/workspace/node1/chain"
NAME = sys.argv[1] if len(sys.argv) > 1 else "health1.txt"

CMD = (
    "cd " + W + "; "
    "{ echo '#### TIME'; date '+%m-%d %H:%M:%S'; "
    "echo '#### PROC_CHAIN(应为1)'; ps -eo pid,etime,cmd | grep '[c]hain_v1.sh'; "
    "echo '#### PROC_TRAIN(应为1组3个)'; ps -eo pid,etime,cmd | grep '[t]rain_remote_v4' | wc -l; "
    "echo '#### DONE'; cat done_steps.txt 2>/dev/null; "
    "echo '#### STOP_FLAG'; if [ -f stop ]; then echo present-BAD; else echo absent-OK; fi; "
    "echo '#### CKPT(权重完整性)'; ls -la ckpt/ | head -12; "
    "echo '-- ckpt config --'; python3 -c \"import json;c=json.load(open('ckpt/config.json'));"
    "print('inter=',c.get('intermediate_size'),'layers=',c.get('num_hidden_layers'),'hidden=',c.get('hidden_size'))\" 2>&1; "
    "echo '#### RUN_TMP(训练输出)'; ls -la run_tmp/checkpoint-latest/ 2>/dev/null | head -8 || echo 'run_tmp 尚未产出'; "
    "echo '#### LOG_TAIL(看是否在推进)'; tail -6 chain.log; "
    "echo '#### CORPUS_R5'; ls -la corpus_r5/; "
    "echo '#### CORPUS_R4'; ls -la corpus_R4/ 2>/dev/null | head -4; "
    "echo '#### 备份区'; ls backup_corpus/ 2>/dev/null; "
    "echo '#### DISK'; df -h /mnt/workspace | tail -1; "
    "echo '#### 0.3B链路就绪'; ls -la chain_v2.sh expand_300m_node1.py 2>/dev/null || echo '尚未落地(等R6后自动拉)'; "
    "} 2>&1 | curl -s -X POST '" + ECS + "/upload?name=" + NAME + "&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "health-" + NAME.replace(".txt", ""),
                   "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
except Exception as e:
    print("POST ERR:", e)

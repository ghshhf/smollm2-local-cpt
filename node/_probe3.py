#!/usr/bin/env python3
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = r"""{ echo '=== 训练相关进程 ==='; pgrep -af 'train_remote|chain_v|node_v5|watchd|maild' | head -8;
echo '=== chain_v2 日志尾 ==='; tail -15 /mnt/workspace/node1/chain/chain_v2.log 2>/dev/null;
echo '=== base_tok 目录 ==='; ls -la /mnt/workspace/node1/chain/base_tok/ 2>/dev/null;
echo '=== heartbeat ==='; cat /mnt/workspace/node1/chain/heartbeat.txt 2>/dev/null;
echo '=== done_steps ==='; cat /mnt/workspace/node1/chain/done_steps.txt; echo;
echo '=== R7 训练日志尾 ==='; ls -t /mnt/workspace/node1/chain/*.log 2>/dev/null | head -3; } 2>&1 | curl -s -m45 -X POST 'http://47.99.131.12:8081/upload?name=probe3_001.txt&key=k0u4mail' --data-binary @-"""
body = json.dumps({"cmd": CMD, "id": "probe3-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

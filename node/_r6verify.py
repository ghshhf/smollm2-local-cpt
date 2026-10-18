#!/usr/bin/env python3
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = r"""{ echo '=== ckpt_old 全部文件 ==='; ls -la /mnt/workspace/node1/chain/ckpt_old/;
echo '=== ckpt_old config 关键字段 ==='; grep -E 'hidden_size|intermediate|num_hidden_layers|vocab' /mnt/workspace/node1/chain/ckpt_old/config.json;
echo '=== ckpt_old sha256 ==='; sha256sum /mnt/workspace/node1/chain/ckpt_old/model.safetensors;
echo '=== chain.log R6 封顶行 ==='; grep -E 'R6|7100|promote|ckpt_old' /mnt/workspace/node1/chain/chain.log 2>/dev/null | tail -8; } 2>&1 | curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=r6_verify_001.txt&key=k0u4mail' --data-binary @-"""
body = json.dumps({"cmd": CMD, "id": "r6-verify-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

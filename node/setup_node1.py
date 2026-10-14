#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""setup_node1.py — 在 node1 部署 E1/V1-02 运行环境：
1) 重组 0.1B（135M 权重 + SmolLM2-200M tokenizer）
2) 把权重与 tokenizer 摆到 eval 脚本默认期望的路径（脚本零改动）
3) 从 ECS /file/ 拉两个 eval 脚本
4) 回传部署清单（setup_done_<ts>.txt）
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
ts = int(time.time())
out_name = f"setup_done_{ts}.txt"

CMD = (
    "cd /mnt/workspace/node1/gh_repo; "
    "B=backups/0.1B/0.1B_extracted; "
    "if [ ! -f \"$B/model/pytorch_model.bin\" ]; then (cd backups/0.1B && python reassemble.py) || { echo REASSEMBLE_FAIL; exit 1; }; fi; "
    "mkdir -p out_smollm2_135m_pt/checkpoint-latest models/SmolLM2-200M eval; "
    "cp -n \"$B/model/pytorch_model.bin\" out_smollm2_135m_pt/checkpoint-latest/pytorch_model.bin; "
    "cp -n \"$B/base/tokenizer.json\" \"$B/base/tokenizer_config.json\" \"$B/base/config.json\" \"$B/base/generation_config.json\" \"$B/base/special_tokens_map.json\" \"$B/base/vocab.json\" \"$B/base/merges.txt\" models/SmolLM2-200M/; "
    "curl -s 'http://47.99.131.12:8081/file/n1_1792001815_e1_subliminal_proxy.py?key=k0u4mail' -o eval/e1_subliminal_proxy.py; "
    "curl -s 'http://47.99.131.12:8081/file/n1_1792001815_v1_02_alpha_ladder.py?key=k0u4mail' -o eval/v1_02_alpha_ladder.py; "
    "{ "
    "echo '=== reassembled ckpt ==='; ls -la out_smollm2_135m_pt/checkpoint-latest/pytorch_model.bin; "
    "echo '=== staged tokenizer ==='; ls models/SmolLM2-200M/; "
    "echo '=== eval scripts ==='; ls -la eval/*.py; "
    "echo '=== transformers ==='; python -c 'import transformers;print(transformers.__version__)' 2>&1 | head -1; "
    "} 2>&1 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=" + out_name + "&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": f"setup-node1-{ts}", "ts": ts}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
    print("fetch:", f"{ECS}/file/{out_name}")
except Exception as e:
    print("POST ERR:", e)

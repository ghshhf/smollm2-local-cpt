#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("{ echo '=== base_tok files ==='; ls -la /mnt/workspace/node1/chain/base_tok/ 2>/dev/null; "
       "echo '=== ckpt files ==='; ls -la /mnt/workspace/node1/chain/ckpt/ 2>/dev/null; "
       "echo '=== ckpt tokenizer_config head ==='; head -c 400 /mnt/workspace/node1/chain/ckpt/tokenizer_config.json 2>/dev/null; echo; "
       "echo '=== tf version ==='; python3 -c 'import transformers; print(transformers.__version__)'; "
       "echo '=== tokenizers lib ==='; python3 -c 'import tokenizers; print(tokenizers.__version__)' 2>&1 | head -2; "
       "echo '=== sentencepiece ==='; python3 -c 'import sentencepiece; print(sentencepiece.__version__)' 2>&1 | head -2; } 2>&1 | "
       "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=tok_diag_001.txt&key=k0u4mail' --data-binary @-")
body = json.dumps({"cmd": CMD, "id": "tok-diag-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

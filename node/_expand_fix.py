#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = ("D=/mnt/workspace/node1/chain; "
       "curl -s -m60 -o $D/expand_300m_node1.py 'http://47.99.131.12:8081/file/expand_300m_node1.py?key=k0u4mail'; "
       "echo \"fetched expand script: $(wc -c < $D/expand_300m_node1.py) bytes\"; "
       "{ echo '=== RUN EXPAND ==='; python3 $D/expand_300m_node1.py $D/ckpt 2>&1 | grep -vE 'Loading|torch_dtype|warnings.warn'; } 2>&1 | tee $D/expand_run.log | tail -8; "
       "RC=${PIPESTATUS[0]}; "
       "if [ -s $D/ckpt/model.safetensors ] && [ \"$(python3 -c \"import json;print(json.load(open('$D/ckpt/config.json'))['intermediate_size'])\")\" = \"4608\" ]; then "
       "  echo 'EXPAND_OK inter=4608'; "
       "  rm -f $D/chain.pid; "
       "  (nohup sh $D/chain_v2.sh >/dev/null 2>&1 </dev/null &); "
       "  sleep 5; echo \"chain_v2 restarted: $(pgrep -af chain_v2 | grep -v grep | head -1)\"; "
       "else echo 'EXPAND_FAILED'; fi; "
       "echo '=== chain_v2.log tail ==='; tail -5 $D/chain_v2.log 2>/dev/null")
body = json.dumps({"cmd": CMD, "id": "expand-fix-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

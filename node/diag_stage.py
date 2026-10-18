#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_stage.py — 诊断为什么 V1_STAGED=0: 看下载文件大小 / dash 语法报错 / 与现网脚本差异。"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
NAME = sys.argv[1] if len(sys.argv) > 1 else "diag1.txt"

CMD = r"""
cd /mnt/workspace/node1/chain
{ echo "#### 残留 .new 文件"; ls -la chain_v1.sh.new chain_v2.sh.new 2>&1 | head -5
  echo "#### 现网脚本大小"; wc -c chain_v1.sh chain_v2.sh 2>&1
  echo "#### 重新下载到 /root (绕开 NAS)"; curl -s -o /root/t1.sh """ + ECS + """/file/node1_chain.sh; echo "curl_rc=$?"; wc -c /root/t1.sh
  echo "#### /bin/sh -n 校验(/root/t1.sh)"; /bin/sh -n /root/t1.sh 2>&1 | head -8; echo "dash_n_rc=$?"
  echo "#### /bin/sh -n 校验(现网 chain_v1.sh)"; /bin/sh -n chain_v1.sh 2>&1 | head -8; echo "rc=$?"
  echo "#### 与现网脚本差异行数"; diff /root/t1.sh chain_v1.sh | wc -l
  echo "#### 首行/末行"; head -1 /root/t1.sh; tail -1 /root/t1.sh
  echo "#### 老链仍在跑?"; ps -eo pid,etime,cmd | grep '[c]hain_v1.sh'; ps -eo pid,cmd | grep '[t]rain_remote_v4' | wc -l
} 2>&1 | curl -s -X POST '""" + ECS + """/upload?name=""" + NAME + """&key=k0u4mail' --data-binary @-
"""

body = json.dumps({"cmd": CMD, "id": "diag-stage", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
except Exception as e:
    print("POST ERR:", e)

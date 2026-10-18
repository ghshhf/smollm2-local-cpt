#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""guard_test.py — 在 node1 上实测守卫的进程识别(单快照祖先回溯版)。
测试脚本故意命名 chain_v9.sh, 让它自己也会被扫到。
预期: FILTERED_OTHER 只剩真正的老链 1921(自身子 shell 全部剔除)。
"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
NAME = sys.argv[1] if len(sys.argv) > 1 else "gtest1.txt"

CMD = r"""
mkdir -p /root/gtest && cd /root/gtest
cat > chain_v9.sh <<'XEOF'
#!/bin/sh
echo "SELF=$$"
other_chain() {
  ps -eo pid,ppid,cmd 2>/dev/null | awk -v self="$$" '
    ($3=="sh"||$3=="dash"||$3=="bash"||$3=="/bin/sh"||$3=="/bin/bash") && $4 ~ /chain_v[0-9]+\.sh$/ {
      if ($1+0 != self+0) key[$1+0] = $2+0
    }
    END {
      for (p in key) {
        q = p; mine = 0; n = 0
        while (n < 25) {
          if (q+0 == self+0) { mine = 1; break }
          if (q in key) q = key[q]; else break
          n++
        }
        if (!mine) printf "%s ", p
      }
    }'
}
echo "RAW(ps 里所有 chain_v*.sh 进程)=[ $(ps -eo pid,ppid,cmd | awk '$3 ~ /sh$/ && $4 ~ /chain_v[0-9]+\.sh$/ {printf "%s(ppid=%s) ", $1, $2}') ]"
echo "FILTERED_OTHER=[ $(other_chain) ]"
echo "RUN2_OTHER=[ $(other_chain) ]"
echo "RUN3_OTHER=[ $(other_chain) ]"
XEOF
sh /root/gtest/chain_v9.sh 2>&1 | curl -s -X POST '""" + ECS + """/upload?name=""" + NAME + """&key=k0u4mail' --data-binary @-
"""

body = json.dumps({"cmd": CMD, "id": "guard-test-" + NAME.replace(".txt", ""),
                   "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
except Exception as e:
    print("POST ERR:", e)

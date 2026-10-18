#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""migrate_v3.py — 把 node1 上正在跑的老链, 无损迁移到加固版 v3。
顺序至关重要(绝不能再犯"先杀后起"的错):
  1) 原子 mv 换脚本   -> 老链持有旧 inode, 完全不受影响
  2) 先起新实例       -> 守卫扫到老链存活, 排队等待(不并发、不碰 run_tmp)
  3) 再 touch stop    -> 老链跑完当前 chunk、promote 完权重、然后才退出(零算力损失)
  4) 老链一退, 新实例 60s 内检测到并接管, 顺手清掉 stop
结果写 mig2.txt。
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
NAME = "mig2.txt"

CMD = r"""
cd /mnt/workspace/node1/chain
curl -s -o chain_v1.sh.new """ + ECS + """/file/node1_chain.sh
curl -s -o chain_v2.sh.new """ + ECS + """/file/chain_v2.sh
V1=0; V2=0
if [ -s chain_v1.sh.new ] && sh -n chain_v1.sh.new 2>/dev/null; then mv chain_v1.sh.new chain_v1.sh; V1=1; fi
if [ -s chain_v2.sh.new ] && sh -n chain_v2.sh.new 2>/dev/null; then mv chain_v2.sh.new chain_v2.sh; V2=1; fi
STARTED=no
if [ "$V1" = "1" ]; then
  (nohup sh chain_v1.sh > /dev/null 2>&1 < /dev/null &)
  sleep 6
  touch stop
  STARTED=yes
fi
{ echo "V1_STAGED=$V1 V2_STAGED=$V2 NEW_STARTED=$STARTED"
  echo "-- ps(应有 老链1921 + 新链等待中) --"
  ps -eo pid,ppid,etime,cmd | grep '[c]hain_v1.sh'
  echo "-- chain.pid --"; cat chain.pid 2>/dev/null || echo none
  echo "-- stop --"; if [ -f stop ]; then echo present; else echo absent; fi
  echo "-- log tail --"; tail -4 chain.log
} 2>&1 | curl -s -X POST '""" + ECS + """/upload?name=""" + NAME + """&key=k0u4mail' --data-binary @-
"""

body = json.dumps({"cmd": CMD, "id": "migrate-v3", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
except Exception as e:
    print("POST ERR:", e)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""migrate_v3b.py — 带自检的无损迁移(老链 -> 加固版 v3)。
顺序与自检:
  1) 原子 mv 换脚本     -> 老链持有旧 inode, 完全不受影响
  2) 先起新实例         -> 守卫扫到老链存活, 排队等待(不并发、不碰 run_tmp)
  3) 等守卫打第一轮日志  -> 校验它只认出 1921 一个 PID
       · 若出现多余 PID(自识别 bug 复发) => 杀掉新实例, 不设 stop, 老链原样继续跑, 迁移自动放弃
       · 若正常 => 才 touch stop, 老链跑完当前 chunk、promote 完权重再退出(零算力损失)
  4) 老链一退, 新实例 60s 内接管, 顺手清掉 stop
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
NAME = "mig6.txt"

CMD = r"""
cd /mnt/workspace/node1/chain
curl -s -o chain_v1.sh.new """ + ECS + """/file/node1_chain.sh
curl -s -o chain_v2.sh.new """ + ECS + """/file/chain_v2.sh
V1=0; V2=0
if [ -s chain_v1.sh.new ] && sh -n chain_v1.sh.new 2>/dev/null; then mv chain_v1.sh.new chain_v1.sh; V1=1; fi
if [ -s chain_v2.sh.new ] && sh -n chain_v2.sh.new 2>/dev/null; then mv chain_v2.sh.new chain_v2.sh; V2=1; fi
RESULT="ABORT_未换脚本"
if [ "$V1" = "1" ]; then
  (nohup sh chain_v1.sh > /dev/null 2>&1 < /dev/null &)
  sleep 100
  LAST=$(grep 'GUARD: 检测到另一条链' chain.log | tail -1)
  if echo "$LAST" | grep -qE 'pid=1921 [0-9]+'; then
    pkill -f '^sh chain_v1.sh' 2>/dev/null
    RESULT="ABORT_守卫自识别复发_已杀新实例_老链继续"
  elif echo "$LAST" | grep -qE 'pid=1921 存活'; then
    touch stop
    RESULT="OK_守卫正常_touch_stop_老链跑完本chunk后优雅退场"
  else
    pkill -f '^sh chain_v1.sh' 2>/dev/null
    RESULT="ABORT_守卫未识别到老链_pattern异常"
  fi
fi
{ echo "V1_STAGED=$V1 V2_STAGED=$V2"
  echo "RESULT=$RESULT"
  echo "-- 守卫最后一行 --"; echo "$LAST"
  echo "-- ps(老链1921 + 新链等待中) --"; ps -eo pid,ppid,etime,cmd | grep '[c]hain_v1.sh' | grep -v '/bin/sh -c'
  echo "-- trainer(应为3) --"; ps -eo pid,cmd | grep '[t]rain_remote_v4' | wc -l
  echo "-- stop --"; if [ -f stop ]; then echo present; else echo absent; fi
  echo "-- log tail --"; tail -3 chain.log
} 2>&1 | curl -s -X POST '""" + ECS + """/upload?name=""" + NAME + """&key=k0u4mail' --data-binary @-
"""

body = json.dumps({"cmd": CMD, "id": "migrate-v3c-" + NAME.replace(".txt",""), "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
except Exception as e:
    print("POST ERR:", e)

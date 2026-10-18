#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""start_watch.py — 在 node1 上拉起训练链看门狗(单实例, 重复执行安全)。
看门狗只在"一条 chain_v*.sh 都没有"时才拉链, 此刻链是活的, 所以本次只启动看门狗本身。
"""
import json, sys, time, urllib.request

ECS = "http://47.99.131.12:8081"
NAME = sys.argv[1] if len(sys.argv) > 1 else "watch1.txt"

CMD = r"""
cd /mnt/workspace/node1/chain
curl -s -o chain_watch.sh """ + ECS + """/file/chain_watch.sh
WR=download_or_syntax_failed
if [ -s chain_watch.sh ] && sh -n chain_watch.sh 2>/dev/null; then
  WN=$(ps -eo pid,cmd | awk '$3 ~ /chain_watch\.sh$/ {n++} END{print n+0}')
  if [ "$WN" -eq 0 ]; then
    (nohup sh chain_watch.sh > /dev/null 2>&1 < /dev/null &)
    WR=started
  else
    WR=already_running
  fi
fi
sleep 4
{ echo "WATCHDOG=$WR"
  echo "-- watchdog procs --"; ps -eo pid,ppid,etime,cmd | grep '[c]hain_watch.sh' | grep -v '/bin/sh -c'
  echo "-- chain procs --"; ps -eo pid,ppid,etime,cmd | grep '[c]hain_v' | grep -v '/bin/sh -c'
  echo "-- stop --"; if [ -f stop ]; then echo present; else echo absent; fi
  echo "-- done --"; cat done_steps.txt 2>/dev/null
  echo "-- watchdog.log --"; tail -3 watchdog.log 2>/dev/null || echo '(无)'
  echo "-- log tail --"; tail -3 chain.log
} 2>&1 | curl -s -X POST '""" + ECS + """/upload?name=""" + NAME + """&key=k0u4mail' --data-binary @-
"""

body = json.dumps({"cmd": CMD, "id": "start-watch-" + NAME.replace(".txt", ""),
                   "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
except Exception as e:
    print("POST ERR:", e)

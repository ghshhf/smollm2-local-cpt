#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mig_undo.py — 止血: 撤掉 stop(保住老链) + 杀掉误判空等的新实例 + 清 pidfile。
新实例只是卡在守卫等待里(没训练、没碰 run_tmp), 杀它零风险; 老链 1921 及其训练器完全不动。
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
NAME = "mig3.txt"

CMD = r"""
cd /mnt/workspace/node1/chain
rm -f stop
pkill -f '^sh chain_v1.sh' 2>/dev/null
sleep 2
rm -f chain.pid
{ echo "STOP: $([ -f stop ] && echo present-BAD || echo absent-OK)"
  echo "-- chain procs (应只剩老链1921) --"
  ps -eo pid,ppid,etime,cmd | grep '[c]hain_v1.sh' | grep -v '/bin/sh -c'
  echo "-- trainer procs (应为3) --"
  ps -eo pid,etime,cmd | grep '[t]rain_remote_v4' | wc -l
  echo "-- chain.pid --"; cat chain.pid 2>/dev/null || echo none
  echo "-- log tail --"; tail -3 chain.log
} 2>&1 | curl -s -X POST '""" + ECS + """/upload?name=""" + NAME + """&key=k0u4mail' --data-binary @-
"""

body = json.dumps({"cmd": CMD, "id": "mig-undo", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200], "| out =", NAME)
except Exception as e:
    print("POST ERR:", e)

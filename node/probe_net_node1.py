#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_net_node1.py — 让 node1 测自身出网能力(它在中国, 能到本机 sandbox 到不了的镜像)
结果上报 ECS /file/status_netprobe.txt
"""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"

CMD = (
    "{ "
    "echo '=== outbound HTTP codes (node1) ==='; "
    "for u in "
    "'https://modelscope.cn' "
    "'https://hf-mirror.com' "
    "'https://hf-mirror.com/datasets/flowerone/chinese-classical-corpus/resolve/main/README.md' "
    "'https://github.com' "
    "'https://raw.githubusercontent.com' "
    "'http://ctext.org' "
    "'https://data.baai.ac.cn' "
    "'https://open.flopsera.com' "
    "; do "
    "  code=$(curl -s -m15 -o /dev/null -w '%{http_code}' \"$u\" 2>/dev/null); "
    "  echo \"$code  $u\"; "
    "done; "
    "echo '=== tools ==='; "
    "which git wget curl python3 2>/dev/null; "
    "echo '=== disk /mnt/workspace ==='; "
    "df -h /mnt/workspace 2>/dev/null | tail -n2; "
    "} | curl -s -X POST 'http://47.99.131.12:8081/upload?name=status_netprobe.txt&key=k0u4mail' --data-binary @-"
)

body = json.dumps({"cmd": CMD, "id": "net-probe", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:200])
except Exception as e:
    print("POST ERR:", e)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = (
    "python3 -c \"import json; d=json.load(open('/mnt/workspace/node1/corpus_04b/poetry/chinese-poetry-collection.json',encoding='utf-8')); "
    "print('type',type(d).__name__); "
    "print('keys',list(d.keys())[:8] if isinstance(d,dict) else 'n/a'); "
    "print('len',len(d)); "
    "print('sample',(d[0] if isinstance(d,list) else d[list(d.keys())[0]][0]) if len(d) else None)\" 2>&1"
)
body = json.dumps({"cmd": CMD, "id": "poetry-diag-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST ->", r.status)
except Exception as e:
    print("ERR:", e)

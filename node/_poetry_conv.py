#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert poetry json to jsonl on node1 + test datasets-server.huggingface.co reachability."""
import json, time, urllib.request

ECS = "http://47.99.131.12:8081"
# 1) poetry -> jsonl (inline python, self-contained)
PY_CONV = (
    "import json;"
    "d=json.load(open('/mnt/workspace/node1/corpus_04b/poetry/chinese-poetry-collection.json',encoding='utf-8'));"
    "n=0;"
    "f=open('/mnt/workspace/node1/corpus_04b/poetry_zh.jsonl','w',encoding='utf-8');"
    "[f.write(json.dumps({'text':r['text1'].strip(),'source':'poetry'},ensure_ascii=False)+chr(10)) or None "
    "for r in d if len((r.get('text1') or '').strip())>=10];"
    "f.close();"
    "print('poetry rows', n if (n:=len(d)) else 0)"
)
CMD1 = (
    "python3 -c \"%s\" 2>&1; wc -l /mnt/workspace/node1/corpus_04b/poetry_zh.jsonl; "
    "du -sh /mnt/workspace/node1/corpus_04b/poetry_zh.jsonl"
) % PY_CONV
body = json.dumps({"cmd": CMD1, "id": "poetry-conv-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("poetry conv POST ->", r.status)
except Exception as e:
    print("poetry conv ERR:", e)

# 2) datasets-server reachability test with回传
CMD2 = (
    "{ echo '--- ds rows test ---'; "
    "curl -s -m 40 'https://datasets-server.huggingface.co/rows?dataset=openbmb%2FUltra-FineWeb-L3&config=multi_style&split=train&offset=0&length=2' | head -c 800; echo; "
    "echo '--- ds size ---'; "
    "curl -s -m 40 'https://datasets-server.huggingface.co/size?dataset=openbmb%2FUltra-FineWeb-L3' | head -c 600; echo; } 2>&1 | "
    "curl -s -X POST 'http://47.99.131.12:8081/upload?name=ds_test_001.txt&key=k0u4mail' --data-binary @-"
)
body2 = json.dumps({"cmd": CMD2, "id": "ds-test-001", "ts": int(time.time())}).encode()
req2 = urllib.request.Request(ECS + "/task", data=body2,
                              headers={"Content-Type": "application/json"}, method="POST")
try:
    r2 = urllib.request.urlopen(req2, timeout=15)
    print("ds test POST ->", r2.status)
except Exception as e:
    print("ds test ERR:", e)

